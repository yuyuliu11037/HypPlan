"""OVM value-model trainer.

Reads the rollouts JSONL produced by `scripts/gen_ovm_rollouts.py` and
trains a scalar `ValueHead` on top of (frozen base + frozen PT-SFT LoRA)
to regress every assistant token toward the rollout's outcome label
in {0, 1}.

Loss: per-token MSE on assistant tokens only (the "Answer:" continuation),
following the OVM paper's convention. Prompt tokens are masked out of
the loss.

Heartbeat: per-rank prints every 60s including step rate, loss, ETA.
Phantom-zero-loss trick to keep DDP in sync if a rank gets a degenerate
batch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
)

from src.ovm_head import ValueHead


def _maybe_init_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = os.environ.get("HYPPLAN_DIST_BACKEND", "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        if backend == "nccl":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend=backend,
                                     init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        return True, rank, world, local_rank, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


class RolloutDataset(Dataset):
    """One sample = one rollout (prompt + generation + outcome label).

    On `__getitem__` we tokenize and return:
      input_ids: (T,)
      attention_mask: (T,)
      label_mask: (T,) — 1 on assistant tokens, 0 on prompt tokens
      outcome: scalar 0 or 1 (broadcast at training time over masked positions)
    """

    def __init__(self, paths: list[str], tok, max_len: int) -> None:
        self.records: list[dict] = []
        for p in paths:
            with open(p) as f:
                for ln in f:
                    r = json.loads(ln)
                    self.records.append(r)
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        prompt = r["prompt"]
        gen = r["generation"]
        # Strip trailing EOS token text if present.
        if gen.endswith(self.tok.eos_token or ""):
            gen = gen[: -len(self.tok.eos_token)]
        full = prompt + gen
        prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
        full_ids = self.tok.encode(full, add_special_tokens=False)
        # If full < prompt (unlikely), bail with all-prompt tokens (label_mask 0).
        if len(full_ids) < len(prompt_ids):
            full_ids = prompt_ids
        # Truncate from left if too long, preserving the assistant tail.
        if len(full_ids) > self.max_len:
            cut = len(full_ids) - self.max_len
            full_ids = full_ids[cut:]
            prompt_ids = prompt_ids[max(0, cut):]   # may shift to 0
        T = len(full_ids)
        prompt_len_in_full = max(0, T - (len(full_ids) - len(prompt_ids)))
        # Simpler: label_mask is 1 for positions after the prompt.
        n_prompt = max(0, T - (len(self.tok.encode(prompt + gen,
                       add_special_tokens=False))
                       - len(self.tok.encode(prompt,
                       add_special_tokens=False))))
        # The above is identical to: positions [len(prompt_ids), T) are
        # assistant tokens. Use that directly.
        label_mask = np.zeros(T, dtype=np.float32)
        # Assistant tokens span [len(prompt_ids), T).
        a_start = min(len(prompt_ids), T)
        label_mask[a_start:] = 1.0
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
            "outcome": float(r["outcome"]),
        }


def _collate(batch, pad_id: int):
    Ts = [b["input_ids"].shape[0] for b in batch]
    Tmax = max(Ts)
    B = len(batch)
    input_ids = torch.full((B, Tmax), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Tmax), dtype=torch.long)
    label_mask = torch.zeros((B, Tmax), dtype=torch.float32)
    outcome = torch.zeros((B,), dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["input_ids"].shape[0]
        input_ids[i, :T] = b["input_ids"]
        attn[i, :T] = 1
        label_mask[i, :T] = b["label_mask"]
        outcome[i] = b["outcome"]
    return {
        "input_ids": input_ids, "attention_mask": attn,
        "label_mask": label_mask, "outcome": outcome,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    is_dist, rank, world, local_rank, device = _maybe_init_dist()
    is_main = (rank == 0)

    if is_main:
        print(f"world={world} device={device}", flush=True)

    base_model = cfg["model"]["base_model"]
    lora_adapter = cfg["model"]["lora_adapter"]
    rollout_paths = cfg["data"]["rollouts"]
    if isinstance(rollout_paths, str):
        rollout_paths = [rollout_paths]
    max_len = int(cfg["data"]["max_seq_len"])
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Loading {base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    if is_main:
        print(f"Attaching LoRA {lora_adapter}", flush=True)
    model = PeftModel.from_pretrained(base, lora_adapter)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = base.config.hidden_size
    head = ValueHead(hidden_dim).to(device)
    head.train()

    if is_dist:
        head = nn.parallel.DistributedDataParallel(
            head, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    if is_main:
        n_train_p = sum(p.numel() for p in head.parameters())
        print(f"head params: {n_train_p}", flush=True)

    # Data.
    ds = RolloutDataset(rollout_paths, tok, max_len)
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank,
                                    shuffle=True, drop_last=True)
                if is_dist else None)

    bs = int(cfg["training"]["batch_size"])
    grad_accum = int(cfg["training"]["grad_accum"])
    epochs = int(cfg["training"]["epochs"])
    pad_id = tok.pad_token_id

    dl = DataLoader(
        ds, batch_size=bs, sampler=sampler, shuffle=(sampler is None),
        collate_fn=lambda b: _collate(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    steps_per_epoch = max(1, len(dl) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.05))
    warmup = max(1, int(total_steps * warmup_ratio))

    opt = AdamW(head.parameters(),
                 lr=float(cfg["training"]["lr"]),
                 weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    if is_main:
        print(f"Train rollouts: {len(ds)}", flush=True)
        print(f"Starting training: total_steps={total_steps}, "
              f"bs={bs}, grad_accum={grad_accum}, world={world}", flush=True)

    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    t0 = time.time()
    last_hb = time.time()
    global_step = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        opt.zero_grad()
        for it, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            label_mask = batch["label_mask"].to(device, non_blocking=True)
            outcome = batch["outcome"].to(device, non_blocking=True)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn,
                            output_hidden_states=True, use_cache=False)
                # last hidden state: (B, T, H)
                h = out.hidden_states[-1]
            h = h.float()
            v = head(h)   # (B, T) in (0,1)
            # MSE on assistant tokens, broadcast outcome over T.
            target = outcome.unsqueeze(1).expand_as(v)
            err2 = (v - target) ** 2
            # Class-imbalance fix: scale loss by per-sample weight if
            # `pos_weight` is set in config. With p_correct=6%, set
            # pos_weight=15 so positive samples contribute roughly the
            # same total gradient as negatives.
            pos_weight = float(cfg["training"].get("pos_weight", 1.0))
            if pos_weight != 1.0:
                w = (1.0 - outcome) + outcome * pos_weight
                err2 = err2 * w.unsqueeze(1)
            denom = label_mask.sum().clamp(min=1.0)
            loss_local = (err2 * label_mask).sum() / denom
            # Phantom-zero-loss safeguard if denom == 0.
            if not torch.isfinite(loss_local):
                loss_local = (v.abs().sum() * 0.0)
            loss = loss_local / grad_accum
            loss.backward()
            if (it + 1) % grad_accum == 0:
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1
            # Per-rank heartbeat.
            now = time.time()
            if now - last_hb >= 60.0:
                rate = (it + 1) / max(now - t0, 1e-6)
                eta_s = (steps_per_epoch * grad_accum * epochs - global_step * grad_accum) / max(rate, 1e-6)
                print(f"[r{rank}] HB epoch={epoch} it={it+1}/{len(dl)} "
                      f"step={global_step}/{total_steps} "
                      f"loss={loss_local.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({(now-t0)/60:.1f}m, eta={eta_s/60:.1f}m)",
                      flush=True)
                last_hb = now
        if is_main:
            print(f"== epoch {epoch} done in {(time.time()-t0):.1f}s ==",
                  flush=True)

    if is_main:
        head_to_save = head.module if is_dist else head
        torch.save({
            "state_dict": head_to_save.state_dict(),
            "config": cfg,
            "hidden_dim": hidden_dim,
        }, output_dir / "ovm_head.pt")
        print(f"Saved ValueHead to {output_dir/'ovm_head.pt'}", flush=True)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
