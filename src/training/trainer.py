from __future__ import annotations

import json
import random
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import save_config
from src.training.eval import evaluate_generation, evaluate_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonlLogger:
    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metrics.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class ExperimentTrainer:
    def __init__(self, model, tokenizer, config: dict[str, Any], output_dir: str | Path) -> None:
        distributed_cfg = config.get("distributed", {})
        mixed_precision = distributed_cfg.get("mixed_precision", "no")
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, self.output_dir)
        self.accelerator.wait_for_everyone()
        self.logger = JsonlLogger(self.output_dir, enabled=self.accelerator.is_main_process)
        self.device = self.accelerator.device

    def _model_module(self):
        return self.accelerator.unwrap_model(self.model)

    def _build_optimizer(self, stage_cfg: dict[str, Any]) -> AdamW:
        return AdamW(
            self._model_module().get_trainable_parameters(),
            lr=stage_cfg["learning_rate"],
            weight_decay=stage_cfg.get("weight_decay", 0.0),
            betas=tuple(stage_cfg.get("betas", [0.9, 0.95])),
            eps=stage_cfg.get("eps", 1e-8),
        )

    def _build_scheduler(self, optimizer: AdamW, stage_cfg: dict[str, Any]):
        max_steps = stage_cfg["max_steps"]
        warmup_steps = stage_cfg.get("warmup_steps", 0)
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

    def _checkpoint_dir(self, stage_name: str) -> Path:
        path = self.output_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(
        self,
        stage_name: str,
        step: int,
        optimizer: AdamW,
        scheduler,
        extra_metrics: dict[str, Any] | None = None,
    ) -> Path:
        self.accelerator.wait_for_everyone()
        checkpoint_dir = self._checkpoint_dir(stage_name)
        state = {
            "step": step,
            "stage_name": stage_name,
            "model_state": self._model_module().trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": extra_metrics or {},
        }
        latest_path = checkpoint_dir / "latest.pt"
        step_path = checkpoint_dir / f"step-{step:06d}.pt"
        if self.accelerator.is_main_process:
            torch.save(state, latest_path)
            torch.save(state, step_path)
        self.accelerator.wait_for_everyone()
        return latest_path

    def load_checkpoint(self, checkpoint_path: str | Path, optimizer: AdamW | None = None, scheduler=None) -> int:
        state = torch.load(checkpoint_path, map_location="cpu")
        self._model_module().load_trainable_state_dict(state["model_state"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        return int(state.get("step", 0))

    def _run_validation(
        self,
        stage_name: str,
        stage_cfg: dict[str, Any],
        val_loader,
        eval_loss_mode: str,
        generation_mode: str | None,
        step: int,
    ) -> dict[str, float]:
        metrics = evaluate_loss(
            model=self._model_module(),
            dataloader=val_loader,
            mode=eval_loss_mode,
            max_batches=stage_cfg.get("eval_loss_batches"),
            accelerator=self.accelerator,
        )

        if generation_mode is not None:
            generation_metrics = evaluate_generation(
                model=self._model_module(),
                tokenizer=self.tokenizer,
                dataloader=val_loader,
                mode=generation_mode,
                max_examples=stage_cfg.get("eval_generation_examples", 0),
                generation_cfg=self.config.get("generation", {}),
                accelerator=self.accelerator,
            )
            metrics.update(generation_metrics)

        record = {"stage": stage_name, "step": step, "split": "validation", **metrics}
        self.logger.write(record)
        self.accelerator.print(record)
        return metrics

    def _run_loop(
        self,
        stage_name: str,
        stage_cfg: dict[str, Any],
        train_loader,
        val_loader,
        compute_loss_fn,
        eval_loss_mode: str,
        generation_mode: str | None,
    ) -> None:
        max_steps = stage_cfg["max_steps"]
        grad_accum_steps = stage_cfg.get("grad_accum_steps", 1)
        log_every = stage_cfg.get("log_every", 10)
        eval_every = stage_cfg.get("eval_every", 0)
        save_every = stage_cfg.get("save_every", 0)
        clip_grad_norm = stage_cfg.get("max_grad_norm", 1.0)

        optimizer = self._build_optimizer(stage_cfg)
        scheduler = self._build_scheduler(optimizer, stage_cfg)

        start_step = 0
        resume_from = stage_cfg.get("resume_from")
        if resume_from:
            start_step = self.load_checkpoint(resume_from, optimizer=optimizer, scheduler=scheduler)

        self.model, optimizer, train_loader, val_loader, scheduler = self.accelerator.prepare(
            self._model_module(),
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )

        train_iterator = cycle(train_loader)
        progress = tqdm(
            range(start_step, max_steps),
            desc=stage_name,
            disable=not self.accelerator.is_local_main_process,
        )
        optimizer.zero_grad(set_to_none=True)

        for step in progress:
            step_start_time = time.time()
            running_weighted_loss = torch.zeros((), device=self.device)
            running_tokens = torch.zeros((), device=self.device, dtype=torch.long)

            for _ in range(grad_accum_steps):
                batch = next(train_iterator)
                metrics = compute_loss_fn(batch)
                loss = metrics["loss"] / grad_accum_steps
                self.accelerator.backward(loss)
                running_weighted_loss = running_weighted_loss + (
                    metrics["loss"].detach() * metrics["token_count"].detach()
                )
                running_tokens = running_tokens + metrics["token_count"].detach()

            if clip_grad_norm is not None and clip_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self._model_module().get_trainable_parameters(), clip_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            current_step = step + 1
            elapsed = time.time() - step_start_time

            if current_step % log_every == 0 or current_step == 1:
                total_weighted_loss = self.accelerator.reduce(running_weighted_loss, reduction="sum")
                total_tokens = self.accelerator.reduce(running_tokens, reduction="sum")
                record = {
                    "stage": stage_name,
                    "step": current_step,
                    "split": "train",
                    "loss": float((total_weighted_loss / total_tokens.clamp_min(1)).item()),
                    "tokens": int(total_tokens.item()),
                    "lr": scheduler.get_last_lr()[0],
                    "step_time_sec": elapsed,
                }
                self.logger.write(record)
                self.accelerator.print(record)

            if eval_every and current_step % eval_every == 0:
                self._run_validation(
                    stage_name=stage_name,
                    stage_cfg=stage_cfg,
                    val_loader=val_loader,
                    eval_loss_mode=eval_loss_mode,
                    generation_mode=generation_mode,
                    step=current_step,
                )

            if save_every and current_step % save_every == 0:
                self.save_checkpoint(
                    stage_name=stage_name,
                    step=current_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        self.save_checkpoint(stage_name=stage_name, step=max_steps, optimizer=optimizer, scheduler=scheduler)
        self.accelerator.wait_for_everyone()
        self.model = self._model_module()

    def train_planning(self, train_loader, val_loader) -> None:
        stage1_cfg = self.config["training"]["stage1"]
        stage2_cfg = self.config["training"]["stage2"]
        model_cfg = self.config["model"]

        if stage1_cfg.get("max_steps", 0) > 0:
            self._model_module().set_stage("stage1")
            self._run_loop(
                stage_name="stage1",
                stage_cfg=stage1_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                compute_loss_fn=self._model_module().compute_stage1_loss,
                eval_loss_mode="planning_stage1",
                generation_mode=None,
            )

        if stage2_cfg.get("max_steps", 0) > 0:
            if model_cfg.get("use_lora", True):
                self._model_module().enable_lora(model_cfg.get("lora", {}))
            self._model_module().set_stage("stage2")
            self._run_loop(
                stage_name="stage2",
                stage_cfg=stage2_cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                compute_loss_fn=self._model_module().compute_stage2_loss,
                eval_loss_mode="planning_stage2",
                generation_mode="planning",
            )

    def train_baseline(self, train_loader, val_loader) -> None:
        stage_cfg = self.config["training"]["baseline"]
        model_cfg = self.config["model"]

        if model_cfg.get("use_lora_for_baseline", True):
            self._model_module().enable_lora(model_cfg.get("lora", {}))

        self._model_module().set_stage("baseline")
        self._run_loop(
            stage_name="baseline",
            stage_cfg=stage_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            compute_loss_fn=self._model_module().compute_baseline_loss,
            eval_loss_mode="baseline",
            generation_mode="baseline",
        )
