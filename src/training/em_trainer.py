from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import (
    AugmentedReasoningPath,
    build_prefix_text,
    initialize_augmented_dataset,
    is_placeholder_dataset_uri,
    load_jsonl_dataset,
    synthetic_reasoning_paths,
)
from src.model.hyperbolic import IdentityProjection
from src.model.planning_head import LoraSettings, PlanningHead, load_tokenizer_and_model
from src.training.losses import compute_plan_loss, compute_reason_loss


@dataclass
class TrainConfig:
    seed: int
    model_name_or_path: str
    dataset_uri: str
    output_dir: str
    num_em_iterations: int
    train_epochs_per_em: int
    max_train_samples: int | None
    max_steps_per_epoch: int | None
    plan_loss_weight: float
    reason_loss_weight: float
    max_question_tokens: int
    max_step_tokens: int
    planning_dim: int
    planning_mlp_hidden_dim: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    per_device_batch_size: int
    mixed_precision: str
    save_every_steps: int
    log_every_steps: int
    use_synthetic_if_placeholder_dataset: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "TrainConfig":
        return TrainConfig(
            seed=cfg["seed"],
            model_name_or_path=cfg["model_name_or_path"],
            dataset_uri=cfg["dataset_uri"],
            output_dir=cfg["output_dir"],
            num_em_iterations=cfg["num_em_iterations"],
            train_epochs_per_em=cfg["train_epochs_per_em"],
            max_train_samples=cfg.get("max_train_samples"),
            max_steps_per_epoch=cfg.get("max_steps_per_epoch"),
            plan_loss_weight=cfg["plan_loss_weight"],
            reason_loss_weight=cfg["reason_loss_weight"],
            max_question_tokens=cfg["max_question_tokens"],
            max_step_tokens=cfg["max_step_tokens"],
            planning_dim=cfg["planning_dim"],
            planning_mlp_hidden_dim=cfg["planning_mlp_hidden_dim"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            max_grad_norm=cfg["max_grad_norm"],
            per_device_batch_size=cfg["per_device_batch_size"],
            mixed_precision=cfg["mixed_precision"],
            save_every_steps=cfg["save_every_steps"],
            log_every_steps=cfg["log_every_steps"],
            use_synthetic_if_placeholder_dataset=cfg["use_synthetic_if_placeholder_dataset"],
            lora_rank=cfg["lora"]["rank"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            lora_target_modules=cfg["lora"]["target_modules"],
        )


class EMTrainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
        )
        torch.manual_seed(cfg.seed)

        lora_settings = LoraSettings(
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        self.tokenizer, self.model = load_tokenizer_and_model(
            model_name_or_path=cfg.model_name_or_path,
            lora_settings=lora_settings,
            device_map=None,
        )
        hidden_size = self.model.config.hidden_size
        self.planning_head = PlanningHead(
            hidden_size=hidden_size,
            planning_dim=cfg.planning_dim,
            mlp_hidden_dim=cfg.planning_mlp_hidden_dim,
        )
        self.projection = IdentityProjection()
        self.plan_to_hidden = nn.Linear(cfg.planning_dim, hidden_size)

        self.optimizer = AdamW(
            list(self.model.parameters())
            + list(self.planning_head.parameters())
            + list(self.plan_to_hidden.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.model, self.planning_head, self.plan_to_hidden, self.optimizer = self.accelerator.prepare(
            self.model, self.planning_head, self.plan_to_hidden, self.optimizer
        )
        self.device = self.accelerator.device

    def load_data(self) -> List[AugmentedReasoningPath]:
        if is_placeholder_dataset_uri(self.cfg.dataset_uri):
            if not self.cfg.use_synthetic_if_placeholder_dataset:
                raise ValueError(
                    "dataset_uri is still placeholder TODO://... . "
                    "Set a real JSONL path or enable use_synthetic_if_placeholder_dataset."
                )
            reasoning_paths = synthetic_reasoning_paths()
            self.accelerator.print("Using synthetic dataset because dataset_uri is placeholder.")
        else:
            reasoning_paths = load_jsonl_dataset(
                dataset_uri=self.cfg.dataset_uri,
                max_samples=self.cfg.max_train_samples,
            )

        return initialize_augmented_dataset(
            reasoning_paths=reasoning_paths,
            planning_dim=self.cfg.planning_dim,
            device=self.device,
        )

    def _prefix_hidden(self, question: str, steps: List[str], step_index: int) -> torch.Tensor:
        prefix_text = build_prefix_text(question=question, steps=steps, step_index=step_index)
        tokenized = self.tokenizer(
            prefix_text,
            truncation=True,
            max_length=self.cfg.max_question_tokens,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        outputs = self.model(
            **tokenized,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
        return hidden

    @torch.no_grad()
    def refresh_planning_latents(self, augmented_data: List[AugmentedReasoningPath]) -> None:
        self.model.eval()
        self.planning_head.eval()
        for sample in tqdm(augmented_data, desc="E-step token refresh", disable=not self.accelerator.is_local_main_process):
            for i in range(len(sample.steps)):
                hidden = self._prefix_hidden(sample.question, sample.steps, i)
                predicted_t = self.projection(self.planning_head(hidden))
                sample.planning_latents[i] = predicted_t.detach()

    def m_step_train_once(self, augmented_data: List[AugmentedReasoningPath], em_iteration: int) -> None:
        self.model.train()
        self.planning_head.train()
        self.plan_to_hidden.train()
        running_plan = 0.0
        running_reason = 0.0
        global_step = 0

        for epoch in range(self.cfg.train_epochs_per_em):
            iterator = tqdm(
                augmented_data,
                desc=f"M-step iter={em_iteration} epoch={epoch}",
                disable=not self.accelerator.is_local_main_process,
            )
            for sample in iterator:
                for i in range(len(sample.steps)):
                    with self.accelerator.accumulate(self.model):
                        hidden = self._prefix_hidden(sample.question, sample.steps, i)
                        pred_t = self.projection(self.planning_head(hidden))
                        target_t = sample.planning_latents[i].detach()

                        loss_plan = compute_plan_loss(pred_t, target_t)
                        loss_reason = compute_reason_loss(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            question=sample.question,
                            steps=sample.steps,
                            step_index=i,
                            plan_latent=target_t,
                            plan_to_hidden=self.plan_to_hidden,
                            max_question_tokens=self.cfg.max_question_tokens,
                            max_step_tokens=self.cfg.max_step_tokens,
                            device=self.device,
                        )

                        loss = (
                            self.cfg.plan_loss_weight * loss_plan
                            + self.cfg.reason_loss_weight * loss_reason
                        )
                        self.accelerator.backward(loss)

                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                list(self.model.parameters())
                                + list(self.planning_head.parameters())
                                + list(self.plan_to_hidden.parameters()),
                                self.cfg.max_grad_norm,
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    running_plan += float(loss_plan.detach().item())
                    running_reason += float(loss_reason.detach().item())
                    global_step += 1

                    if global_step % self.cfg.log_every_steps == 0:
                        avg_plan = running_plan / self.cfg.log_every_steps
                        avg_reason = running_reason / self.cfg.log_every_steps
                        self.accelerator.print(
                            f"[iter={em_iteration}] step={global_step} "
                            f"loss_plan={avg_plan:.4f} loss_reason={avg_reason:.4f}"
                        )
                        running_plan = 0.0
                        running_reason = 0.0

                    if global_step % self.cfg.save_every_steps == 0:
                        self.save_checkpoint(em_iteration=em_iteration, global_step=global_step)

                    if self.cfg.max_steps_per_epoch and global_step >= self.cfg.max_steps_per_epoch:
                        return

    def save_checkpoint(self, em_iteration: int, global_step: int) -> None:
        out_dir = Path(self.cfg.output_dir) / f"em_iter_{em_iteration}_step_{global_step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(out_dir / "lora_adapter")
        self.tokenizer.save_pretrained(out_dir / "tokenizer")
        torch.save(
            self.accelerator.unwrap_model(self.planning_head).state_dict(),
            out_dir / "planning_head.pt",
        )
        torch.save(
            self.accelerator.unwrap_model(self.plan_to_hidden).state_dict(),
            out_dir / "plan_to_hidden.pt",
        )
        self.accelerator.print(f"Saved checkpoint to {out_dir}")

    def train(self) -> None:
        augmented = self.load_data()
        self.accelerator.print("Initializing planning latents (Step 1).")
        self.refresh_planning_latents(augmented)

        for em_iter in range(1, self.cfg.num_em_iterations + 1):
            self.accelerator.print(f"Starting EM iteration {em_iter}/{self.cfg.num_em_iterations}.")
            self.m_step_train_once(augmented, em_iteration=em_iter)
            self.accelerator.print(f"Refreshing planning latents after M-step (E-step), iter={em_iter}.")
            self.refresh_planning_latents(augmented)
            self.save_checkpoint(em_iteration=em_iter, global_step=0)
