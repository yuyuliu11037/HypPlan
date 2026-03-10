from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import math
import sys

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import (
    AugmentedReasoningPath,
    build_prefix_text,
    get_rank_shard_indices,
    initialize_augmented_dataset,
    load_gsm8k_aug_dataset,
)
from src.model.hyperbolic import IdentityProjection
from src.model.planning_head import LoraSettings, PlanningHead, load_tokenizer_and_model
from src.training.losses import compute_plan_loss, compute_reason_loss_batch


@dataclass
class TrainConfig:
    seed: int
    model_name_or_path: str
    dataset_name: str
    train_split: str
    eval_split: str
    output_dir: str
    num_em_iterations: int
    train_epochs_per_em: int
    max_train_samples: int | None
    train_subset_ratio: float
    train_subset_seed: int
    max_steps_per_epoch: int | None
    plan_loss_weight: float
    reason_loss_weight: float
    max_question_tokens: int
    max_step_tokens: int
    planning_dim: int | None
    planning_mlp_hidden_dim: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    per_device_batch_size: int
    mixed_precision: str
    save_every_steps: int
    log_every_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "TrainConfig":
        return TrainConfig(
            seed=cfg["seed"],
            model_name_or_path=cfg["model_name_or_path"],
            dataset_name=cfg["dataset_name"],
            train_split=cfg.get("train_split", "train"),
            eval_split=cfg.get("eval_split", "validation"),
            output_dir=cfg["output_dir"],
            num_em_iterations=cfg["num_em_iterations"],
            train_epochs_per_em=cfg["train_epochs_per_em"],
            max_train_samples=cfg.get("max_train_samples"),
            train_subset_ratio=float(cfg.get("train_subset_ratio", 1.0)),
            train_subset_seed=int(cfg.get("train_subset_seed", cfg["seed"])),
            max_steps_per_epoch=cfg.get("max_steps_per_epoch"),
            plan_loss_weight=cfg["plan_loss_weight"],
            reason_loss_weight=cfg["reason_loss_weight"],
            max_question_tokens=cfg["max_question_tokens"],
            max_step_tokens=cfg["max_step_tokens"],
            planning_dim=cfg.get("planning_dim"),
            planning_mlp_hidden_dim=cfg["planning_mlp_hidden_dim"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            max_grad_norm=cfg["max_grad_norm"],
            per_device_batch_size=cfg["per_device_batch_size"],
            mixed_precision=cfg["mixed_precision"],
            save_every_steps=cfg["save_every_steps"],
            log_every_steps=cfg["log_every_steps"],
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
        self.hidden_size = hidden_size
        self.planning_head = PlanningHead(
            hidden_size=hidden_size,
            planning_dim=hidden_size,
            mlp_hidden_dim=cfg.planning_mlp_hidden_dim,
        )
        if cfg.planning_dim is not None and cfg.planning_dim != hidden_size and self.accelerator.is_main_process:
            self.accelerator.print(
                f"Warning: planning_dim={cfg.planning_dim} ignored; using model hidden_size={hidden_size}."
            )
        self.projection = IdentityProjection()

        self.optimizer = AdamW(
            list(self.model.parameters())
            + list(self.planning_head.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.model, self.planning_head, self.optimizer = self.accelerator.prepare(
            self.model, self.planning_head, self.optimizer
        )
        self.device = self.accelerator.device
        self.num_processes = self.accelerator.num_processes
        self.process_index = self.accelerator.process_index

    def load_data(self) -> List[AugmentedReasoningPath]:
        if not (0.0 < self.cfg.train_subset_ratio <= 1.0):
            raise ValueError(
                f"train_subset_ratio must be in (0, 1], got {self.cfg.train_subset_ratio}"
            )
        reasoning_paths = load_gsm8k_aug_dataset(
            dataset_name=self.cfg.dataset_name,
            split=self.cfg.train_split,
            max_samples=self.cfg.max_train_samples,
        )   # reasoning_paths[0]: ReasoningPath(sample_id='train-0', question='Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?', steps=['<<600*30/100=180>>', '<<600*10/100=60>>', '<<180+60=240>>', '<<600-240=360>>'], answer='360')
        total_loaded = len(reasoning_paths)
        if self.cfg.train_subset_ratio < 1.0 and total_loaded > 0:
            subset_size = max(1, int(total_loaded * self.cfg.train_subset_ratio))
            if subset_size < total_loaded:
                generator = torch.Generator().manual_seed(self.cfg.train_subset_seed)
                indices = torch.randperm(total_loaded, generator=generator)[:subset_size].tolist()
                reasoning_paths = [reasoning_paths[idx] for idx in indices]
            if self.accelerator.is_main_process:
                self.accelerator.print(
                    "Applying train subset: "
                    f"ratio={self.cfg.train_subset_ratio:.4f}, "
                    f"seed={self.cfg.train_subset_seed}, "
                    f"samples={len(reasoning_paths)}/{total_loaded}"
                )

        return initialize_augmented_dataset(
            reasoning_paths=reasoning_paths,
            planning_dim=self.hidden_size,
        )

    def _rank_indices(self, total_size: int) -> List[int]:
        return get_rank_shard_indices(
            total_size=total_size,
            process_index=self.process_index,
            num_processes=self.num_processes,
        )

    def _prefix_hidden(self, question: str, steps: List[str], step_index: int) -> torch.Tensor:
        prefix_text = build_prefix_text(question=question, steps=steps, step_index=step_index)
        return self._prefix_hidden_batch([prefix_text]).squeeze(0)

    def _prefix_hidden_batch(self, prefix_texts: List[str]) -> torch.Tensor:
        if not prefix_texts:
            raise ValueError("prefix_texts must be non-empty")
        tokenized = self.tokenizer(
            prefix_texts,
            truncation=True,
            max_length=self.cfg.max_question_tokens,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        outputs = self.model(
            **tokenized,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        last_token_idx = tokenized["attention_mask"].sum(dim=1).sub(1).clamp_min(0)
        batch_idx = torch.arange(hidden_states.size(0), device=self.device)
        hidden = hidden_states[batch_idx, last_token_idx, :]
        return hidden

    @torch.no_grad()
    def refresh_planning_latents(self, augmented_data: List[AugmentedReasoningPath]) -> None:
        self.model.eval()
        self.planning_head.eval()
        shard_indices = self._rank_indices(len(augmented_data))
        local_updates: List[tuple[int, List[torch.Tensor]]] = []
        local_entries: List[tuple[int, int]] = []
        for sample_idx in shard_indices:
            sample = augmented_data[sample_idx]
            for step_idx in range(len(sample.steps)):
                local_entries.append((sample_idx, step_idx))
        batch_size = max(1, self.cfg.per_device_batch_size)
        sample_cache: Dict[int, List[torch.Tensor | None]] = {
            sample_idx: [None] * len(augmented_data[sample_idx].steps) for sample_idx in shard_indices
        }

        iterator = tqdm(
            range(0, len(local_entries), batch_size),
            desc=f"E-step token refresh rank={self.process_index}",
            disable=not self.accelerator.is_local_main_process,
        )
        for batch_start in iterator:
            batch_entries = local_entries[batch_start : batch_start + batch_size]
            prefix_texts = []
            for sample_idx, step_idx in batch_entries:
                sample = augmented_data[sample_idx]
                prefix_texts.append(build_prefix_text(sample.question, sample.steps, step_idx))
            hidden = self._prefix_hidden_batch(prefix_texts)
            # Future Lorentz constraints should be enforced after projection and before caching.
            predicted_t = self.projection(self.planning_head(hidden))
            predicted_cpu = predicted_t.detach().to("cpu", dtype=torch.float32)
            for row_idx, (sample_idx, step_idx) in enumerate(batch_entries):
                sample_cache[sample_idx][step_idx] = predicted_cpu[row_idx]

        for sample_idx in shard_indices:
            cached = sample_cache[sample_idx]
            latents: List[torch.Tensor] = []
            for latent in cached:
                if latent is None:
                    raise RuntimeError("Missing cached latent during E-step refresh")
                latents.append(latent)
            local_updates.append((sample_idx, latents))

        # Each rank only trains on its local shard, so local refresh is sufficient.
        # Avoid all_gather_object on large Python payloads, which can OOM GPU buffers.
        for sample_idx, latents in local_updates:
            augmented_data[sample_idx].planning_latents = latents
        self.accelerator.wait_for_everyone()

    def m_step_train_once(self, augmented_data: List[AugmentedReasoningPath], em_iteration: int) -> None:
        self.model.train()
        self.planning_head.train()
        shard_indices = self._rank_indices(len(augmented_data))
        local_entries: List[tuple[int, int]] = []
        for sample_idx in shard_indices:
            sample = augmented_data[sample_idx]
            for step_idx in range(len(sample.steps)):
                local_entries.append((sample_idx, step_idx))
        batch_size = max(1, self.cfg.per_device_batch_size)
        local_num_batches = math.ceil(len(local_entries) / batch_size) if local_entries else 0
        local_num_batches_tensor = torch.tensor([local_num_batches], device=self.device, dtype=torch.int64)
        gathered_num_batches = self.accelerator.gather(local_num_batches_tensor)
        max_num_batches = int(gathered_num_batches.max().item())
        if self.cfg.max_steps_per_epoch is not None:
            max_num_batches = min(max_num_batches, self.cfg.max_steps_per_epoch)

        dummy_entry: tuple[int, int] | None = None
        if max_num_batches > 0:
            if local_entries:
                dummy_entry = local_entries[0]
            else:
                for sample_idx, sample in enumerate(augmented_data):
                    if sample.steps:
                        dummy_entry = (sample_idx, 0)
                        break
                if dummy_entry is None:
                    raise RuntimeError("No valid reasoning steps found for dummy synchronized batches.")
        local_steps = 0
        running_plan = 0.0
        running_reason = 0.0
        running_count = 0.0
        total_plan = 0.0
        total_reason = 0.0
        total_count = 0.0

        for epoch in range(self.cfg.train_epochs_per_em):
            iterator = tqdm(
                range(max_num_batches),
                desc=f"M-step iter={em_iteration} epoch={epoch} rank={self.process_index}",
                disable=not self.accelerator.is_local_main_process,
            )
            for batch_idx in iterator:
                batch_start = batch_idx * batch_size
                is_real_batch = batch_start < len(local_entries)
                if is_real_batch:
                    batch_entries = local_entries[batch_start : batch_start + batch_size]
                else:
                    if dummy_entry is None:
                        raise RuntimeError("Missing dummy entry for synchronized DDP step.")
                    batch_entries = [dummy_entry]
                with self.accelerator.accumulate(self.model):
                    questions: List[str] = []
                    steps_batch: List[List[str]] = []
                    step_indices: List[int] = []
                    prefix_texts: List[str] = []
                    target_latents: List[torch.Tensor] = []
                    for sample_idx, step_idx in batch_entries:
                        sample = augmented_data[sample_idx]
                        questions.append(sample.question)
                        steps_batch.append(sample.steps)
                        step_indices.append(step_idx)
                        prefix_texts.append(build_prefix_text(sample.question, sample.steps, step_idx))
                        target_latents.append(sample.planning_latents[step_idx])

                    hidden = self._prefix_hidden_batch(prefix_texts)
                    # Keep projection in the predictive path so plan loss trains projected latents.
                    pred_t = self.projection(self.planning_head(hidden))
                    target_t = torch.stack(target_latents, dim=0).to(self.device).detach()

                    loss_plan = compute_plan_loss(pred_t, target_t)
                    loss_reason = compute_reason_loss_batch(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        questions=questions,
                        steps_batch=steps_batch,
                        step_indices=step_indices,
                        plan_latents=target_t,
                        max_question_tokens=self.cfg.max_question_tokens,
                        max_step_tokens=self.cfg.max_step_tokens,
                        device=self.device,
                    )
                    if not is_real_batch:
                        # Keep collective order aligned across ranks without updating weights.
                        loss_plan = loss_plan * 0.0
                        loss_reason = loss_reason * 0.0

                    loss = (
                        self.cfg.plan_loss_weight * loss_plan
                        + self.cfg.reason_loss_weight * loss_reason
                    )
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            list(self.model.parameters())
                            + list(self.planning_head.parameters()),
                            self.cfg.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if is_real_batch:
                    running_plan += float(loss_plan.detach().item())
                    running_reason += float(loss_reason.detach().item())
                    running_count += 1.0
                    total_plan += float(loss_plan.detach().item())
                    total_reason += float(loss_reason.detach().item())
                    total_count += 1.0
                    local_steps += 1

                if is_real_batch and local_steps % self.cfg.log_every_steps == 0:
                    if self.accelerator.is_main_process:
                        denom = max(running_count, 1.0)
                        avg_plan = running_plan / denom
                        avg_reason = running_reason / denom
                        self.accelerator.print(
                            f"[iter={em_iteration}] local_step={local_steps} "
                            f"rank0_running_loss_plan={avg_plan:.4f} rank0_running_loss_reason={avg_reason:.4f}"
                        )
                    running_plan = 0.0
                    running_reason = 0.0
                    running_count = 0.0

                if is_real_batch and local_steps > 0 and local_steps % self.cfg.save_every_steps == 0:
                    self.save_checkpoint(em_iteration=em_iteration, global_step=local_steps)

        total_stats = torch.tensor(
            [total_plan, total_reason, total_count],
            device=self.device,
            dtype=torch.float32,
        )
        reduced_totals = self.accelerator.reduce(total_stats, reduction="sum")
        if self.accelerator.is_main_process:
            denom = max(float(reduced_totals[2].item()), 1.0)
            self.accelerator.print(
                f"[iter={em_iteration}] global_epoch_loss_plan={float(reduced_totals[0].item())/denom:.4f} "
                f"global_epoch_loss_reason={float(reduced_totals[1].item())/denom:.4f}"
            )
        self.accelerator.wait_for_everyone()

    def save_checkpoint(self, em_iteration: int, global_step: int) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        out_dir = Path(self.cfg.output_dir) / f"em_iter_{em_iteration}_step_{global_step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(out_dir / "lora_adapter")
        self.tokenizer.save_pretrained(out_dir / "tokenizer")
        torch.save(
            self.accelerator.unwrap_model(self.planning_head).state_dict(),
            out_dir / "planning_head.pt",
        )
        self.accelerator.print(f"Saved checkpoint to {out_dir}")

    def train(self) -> None:
        augmented = self.load_data()   # augmented[0]: sample_id='train-0', question='Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?', steps=['<<600*30/100=180>>', '<<600*10/100=60>>', '<<180+60=240>>', '<<600-240=360>>'], answer='360', planning_latents=[tensor([0., 0., 0.,  ..., 0., 0., 0.]), tensor([0., 0., 0.,  ..., 0., 0., 0.]), tensor([0., 0., 0.,  ..., 0., 0., 0.]), tensor([0., 0., 0.,  ..., 0., 0., 0.])]
        local_shard = self._rank_indices(len(augmented))
        if self.accelerator.is_main_process:
            effective_batch = (
                self.cfg.per_device_batch_size
                * self.cfg.gradient_accumulation_steps
                * self.num_processes
            )
            self.accelerator.print(
                "DDP runtime: "
                f"world_size={self.num_processes}, "
                f"effective_batch_estimate={effective_batch}, "
                f"rank0_local_samples={len(local_shard)}, "
                f"total_samples={len(augmented)}"
            )
        self.accelerator.print("Initializing planning latents (Step 1).")
        self.refresh_planning_latents(augmented)

        for em_iter in range(1, self.cfg.num_em_iterations + 1):
            self.accelerator.print(f"Starting EM iteration {em_iter}/{self.cfg.num_em_iterations}.")
            self.m_step_train_once(augmented, em_iteration=em_iter)
            self.accelerator.print(f"Refreshing planning latents after M-step (E-step), iter={em_iter}.")
            self.refresh_planning_latents(augmented)
            self.save_checkpoint(em_iteration=em_iter, global_step=0)
