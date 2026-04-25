"""Qwen-14B SFT on Game-24 (bf16 DDP LoRA) — token-matched baseline for DAgger.

Uses `fewshot_chat_prompt_24` so the SFT model trains under the same prompt
distribution that DAgger stage-2 uses. Optimiser (AdamW lr 1e-4, cosine,
grad_clip 1.0, bf16) and LoRA config (r=16, α=32, q/k/v/o_proj) match
DAgger's stage-2 config so any accuracy gap reflects methodology, not
hyperparams.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.dataset_24 import Game24SFTChatDataset, collate_fn


def _setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        from datetime import timedelta
        backend = os.environ.get("HYPPLAN_DIST_BACKEND", "gloo")
        dist.init_process_group(backend=backend, timeout=timedelta(hours=8))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return distributed, device, rank, world_size, local_rank


def _manual_all_reduce_grads(params, world_size: int) -> None:
    """Match the DAgger DDP convention: average grads across ranks manually."""
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, device, rank, world_size, _ = _setup_distributed()
    torch.manual_seed(args.seed + rank)

    model_name = config["model"]["base_model"]
    if rank == 0:
        print(f"Loading {model_name} in bf16...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if rank == 0:
        model.print_trainable_parameters()

    ds = Game24SFTChatDataset(
        tokenizer, config["data"]["train_data"],
        max_seq_len=config["data"].get("max_seq_len", 800),
        prompt_style=config["data"].get("prompt_style", "fewshot"),
        unique_problems=config["data"].get("unique_problems", True),
    )

    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed,
        )
    loader = DataLoader(
        ds, batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable,
                       lr=float(config["training"]["lr"]),
                       weight_decay=float(config["training"].get("weight_decay", 0.0)))

    epochs = int(config["training"]["epochs"])
    grad_accum = int(config["training"].get("grad_accum", 1))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    total_steps = (len(loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * float(config["training"].get("warmup_ratio", 0.05)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    out_dir = config["training"]["output_dir"]
    log_dir = config["training"].get("log_dir", "logs")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
    log_path = os.path.join(log_dir, "sft_24_qwen_train.jsonl")

    if rank == 0:
        print(f"Training: {len(ds)} samples, {epochs} epochs, "
              f"bs={config['training']['batch_size']}/gpu, "
              f"world={world_size}, grad_accum={grad_accum}, "
              f"total_optim_steps={total_steps}", flush=True)

    global_step = 0
    model.train()
    t0 = time.time()

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        accum_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids,
                         attention_mask=attn_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += float(loss.item())

            if (batch_idx + 1) % grad_accum == 0:
                if distributed:
                    _manual_all_reduce_grads(trainable, world_size)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % 10 == 0:
                    print(f"  step {global_step}/{total_steps} | "
                          f"epoch {epoch} | loss={accum_loss:.4f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "epoch": epoch,
                            "loss": round(accum_loss, 4),
                            "lr": scheduler.get_last_lr()[0],
                        }) + "\n")
                accum_loss = 0.0

        if rank == 0:
            print(f"=== epoch {epoch} complete in {time.time()-t0:.0f}s ===",
                  flush=True)

    if rank == 0:
        lora_dir = os.path.join(out_dir, "lora")
        os.makedirs(lora_dir, exist_ok=True)
        model.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)
        print(f"Saved LoRA to {lora_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
