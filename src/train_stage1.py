"""Stage 1: Warm up Proj with frozen LLM."""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.dataset import MathDataset, collate_fn
from src.model import HypPlanModel


def train_stage1(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        rank = torch.distributed.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    is_main = rank == 0

    # Build model
    model = HypPlanModel(config)
    model.freeze_base_model()
    model.to(torch.bfloat16).to(device)

    if distributed:
        model.proj = torch.nn.parallel.DistributedDataParallel(
            model.proj, device_ids=[local_rank]
        )

    # Dataset
    dataset = MathDataset(
        tokenizer=model.tokenizer,
        split="train",
        max_seq_len=config["data"]["max_seq_len"],
        configs=config["data"]["configs"],
    )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config["stage1"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    trainable_params = list(model.proj.parameters())
    optimizer = AdamW(trainable_params, lr=config["stage1"]["lr"], weight_decay=0.01)

    grad_accum = config["stage1"]["grad_accum"]
    epochs = config["stage1"]["epochs"]
    total_steps = (len(loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * config["stage1"].get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        print(f"Stage 1: {len(dataset)} samples, {len(loader)} batches/epoch")
        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    # Training loop
    model.base_model.eval()
    model.proj.train()

    global_step = 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            boundary_pos = batch["boundary_positions"].to(device)
            inject_pos = batch["inject_positions"].to(device)

            loss, _ = model.forward_stage1(
                input_ids, attention_mask, labels, boundary_pos, inject_pos
            )

            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 50 == 0:
                    print(
                        f"  Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {loss.item() * grad_accum:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

            epoch_loss += loss.item() * grad_accum
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if is_main:
            print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    if is_main:
        output_dir = config["stage1"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        proj_state = model.proj.module.state_dict() if distributed else model.proj.state_dict()
        torch.save({
            "proj": proj_state,
        }, os.path.join(output_dir, "checkpoint.pt"))
        model.tokenizer.save_pretrained(output_dir)
        print(f"Saved Stage 1 checkpoint to {output_dir}")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_stage1(args.config)
