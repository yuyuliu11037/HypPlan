"""CoT-SFT baseline: standard LoRA fine-tuning without planning tokens."""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.dataset_stage1 import Stage1Dataset, collate_stage1
from src.model.lora_utils import setup_lora


def train_cot_sft(config_path: str, local_rank: int = -1):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cfg = config["cot_sft"]

    # Distributed setup — torchrun sets LOCAL_RANK env var
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
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

    # Load tokenizer
    model_name = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = setup_lora(
        model,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )
        _model = model.module
    else:
        _model = model

    # Dataset — correct generations, no [PLAN] tokens
    dataset = Stage1Dataset(
        data_path=config["data"]["math_filtered"],
        tokenizer=tokenizer,
        max_seq_len=config["data"]["max_seq_len"],
        step_delimiter=config["data"]["step_delimiter"],
        insert_plan_token=False,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_stage1,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg["lr"], weight_decay=0.01)

    grad_accum = cfg["grad_accum"]
    epochs = cfg["epochs"]
    total_steps = (len(loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        n_params = sum(p.numel() for p in trainable_params)
        print(f"CoT-SFT: {len(dataset)} samples, {len(loader)} batches/epoch")
        print(f"Trainable params: {n_params:,}")
        print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    # Training loop
    model.train()
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Compute loss with label masking (same as HypPlan stages)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 50 == 0:
                    print(
                        f"  Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if is_main:
            print(f"Epoch {epoch+1}/{epochs} — Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    if is_main:
        output_dir = cfg["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        _model.save_pretrained(os.path.join(output_dir, "lora_adapters"))
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        print(f"Saved CoT-SFT checkpoint to {output_dir}")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train_cot_sft(args.config, args.local_rank)
