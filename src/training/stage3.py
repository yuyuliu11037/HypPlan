"""Stage 3: Joint fine-tuning with LoRA + two-pass planning vector injection."""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset_stage1 import Stage1Dataset, collate_stage1
from src.model.lora_utils import setup_lora
from src.model.plan_model import HypPlanModel


def train_stage3(config_path: str, local_rank: int = -1):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Distributed setup
    distributed = local_rank >= 0
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    is_main = rank == 0

    # Build model
    model = HypPlanModel(config)

    # Load Stage 2 checkpoint for Proj
    stage2_ckpt_path = os.path.join(config["stage2"]["output_dir"], "checkpoint.pt")
    if os.path.exists(stage2_ckpt_path):
        ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=True)
        model.proj.load_state_dict(ckpt["proj"])
        model.project_back.load_state_dict(ckpt["project_back"])
        if is_main:
            print(f"Loaded Stage 2 checkpoint from {stage2_ckpt_path}")
    else:
        if is_main:
            print("WARNING: No Stage 2 checkpoint found")

    # Apply LoRA to base model
    stage3_cfg = config["stage3"]
    model.base_model = setup_lora(
        model.base_model,
        lora_r=stage3_cfg["lora_r"],
        lora_alpha=stage3_cfg["lora_alpha"],
        target_modules=stage3_cfg["lora_target_modules"],
    )

    # Freeze [PLAN] token embedding and lm_head
    model.freeze_plan_token_embedding()
    for param in model.base_model.base_model.lm_head.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing
    model.base_model.gradient_checkpointing_enable()

    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        _model = model.module
    else:
        _model = model

    # Dataset — correct generations with [PLAN] tokens
    dataset = Stage1Dataset(
        data_path=config["data"]["math_filtered"],
        tokenizer=_model.tokenizer,
        max_seq_len=config["data"]["max_seq_len"],
        step_delimiter=config["data"]["step_delimiter"],
        insert_plan_token=True,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=stage3_cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_stage1,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer: LoRA params + Proj + ProjectBack
    trainable_params = [
        {"params": [p for p in _model.base_model.parameters() if p.requires_grad],
         "lr": stage3_cfg["lr"]},
        {"params": list(_model.proj.parameters()) + list(_model.project_back.parameters()),
         "lr": stage3_cfg["lr"]},
    ]
    optimizer = AdamW(trainable_params, weight_decay=0.01)

    grad_accum = stage3_cfg["grad_accum"]
    epochs = stage3_cfg["epochs"]
    total_steps = (len(loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * stage3_cfg.get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        lora_params = sum(p.numel() for p in _model.base_model.parameters() if p.requires_grad)
        proj_params = sum(p.numel() for p in _model.proj.parameters()) + \
                      sum(p.numel() for p in _model.project_back.parameters())
        print(f"Stage 3: {len(dataset)} samples, {len(loader)} batches/epoch")
        print(f"LoRA params: {lora_params:,}, Proj params: {proj_params:,}")
        print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    # Training loop
    _model.base_model.train()
    _model.proj.train()
    _model.project_back.train()

    global_step = 0
    plan_correct_total = 0
    plan_total = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            plan_positions = batch.get("plan_positions")

            if plan_positions is None or plan_positions.numel() == 0:
                continue
            plan_positions = plan_positions.to(device)

            # Inject positions: first token after each [PLAN]
            inject_positions = (plan_positions + 1).clamp(max=input_ids.size(1) - 1)
            inject_positions = inject_positions.where(
                plan_positions >= 0,
                torch.tensor(-1, device=device),
            )

            # Pass 1: collect planning vectors (no grad through LLM hidden states)
            t, z, valid_mask = _model.forward_stage3_pass1(
                input_ids, attention_mask, plan_positions
            )

            # Pass 2: inject and compute loss
            loss = _model.forward_stage3_pass2(
                input_ids, attention_mask, labels,
                z, inject_positions, valid_mask
            )

            # Monitor [PLAN] prediction accuracy
            with torch.no_grad():
                shift_logits = _model.inject_plan_vectors(
                    input_ids, z, inject_positions, valid_mask, attention_mask
                ).logits
                for b in range(input_ids.size(0)):
                    for s in range(plan_positions.size(1)):
                        pp = plan_positions[b, s].item()
                        if pp > 0 and pp < input_ids.size(1):
                            pred = shift_logits[b, pp - 1].argmax().item()
                            plan_total += 1
                            if pred == _model.plan_token_id:
                                plan_correct_total += 1

            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in trainable_params for p in pg["params"]], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 50 == 0:
                    plan_acc = plan_correct_total / max(plan_total, 1)
                    print(
                        f"  Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {loss.item() * grad_accum:.4f} | "
                        f"[PLAN] acc: {plan_acc:.3f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    plan_correct_total = 0
                    plan_total = 0

            epoch_loss += loss.item() * grad_accum
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if is_main:
            print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    if is_main:
        output_dir = stage3_cfg["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapters
        _model.base_model.save_pretrained(os.path.join(output_dir, "lora_adapters"))

        # Save Proj and ProjectBack
        torch.save({
            "proj": _model.proj.state_dict(),
            "project_back": _model.project_back.state_dict(),
        }, os.path.join(output_dir, "proj_checkpoint.pt"))

        # Save tokenizer
        _model.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        print(f"Saved Stage 3 checkpoint to {output_dir}")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train_stage3(args.config, args.local_rank)
