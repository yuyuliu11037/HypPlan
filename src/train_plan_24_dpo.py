"""DPO-style preference training for planning vectors on Game of 24.

For each preference pair (ctx, r+, r-):
  z = ProjMLP(h_boundary)
  log π(r+ | ctx, z) via frozen base with z injected before r+
  log π(r- | ctx, z) via frozen base with z injected before r-
  log π_ref(r+ | ctx) and log π_ref(r- | ctx) are precomputed (no z).

  L_DPO(z) = -log σ( β·[(logp_pos_with - logp_pos_ref) - (logp_neg_with - logp_neg_ref)] )

Gradient flows only through z (the ProjMLP). Supports DDP with NCCL_P2P_DISABLE=1.
Monitors ‖z‖ (mean/std/min/max) to catch explosion or collapse early.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from src.projections import ProjMLP
from src.dataset_24_dpo import Game24DPODataset, collate_fn


def seq_logprob_with_z(base_model, embed_table, ctx_ids, z, tail_ids, device):
    """Log-prob sum of tail_ids under frozen base with z injected between ctx and tail.

    Builds inputs_embeds = [embed(ctx), z, embed(tail)].
    Logits at position (ctx_len) predict z-token (ignored). Logits at positions
    (ctx_len+1 .. ctx_len+tail_len) predict tail_ids token by token.

    Specifically: position p's logits predict token at p+1 of inputs_embeds.
      Tail tokens are at positions (ctx_len+1 .. ctx_len+tail_len).
      The logits that predict them are at positions (ctx_len .. ctx_len+tail_len-1).
    Returns scalar tensor (grad flows through z).
    """
    with torch.no_grad():
        ctx_embeds = embed_table(ctx_ids)
        tail_embeds = embed_table(tail_ids)
    embed_dtype = ctx_embeds.dtype

    # z: (H,) float → cast to embed dtype, unsqueeze to (1, H)
    z_cast = z.to(embed_dtype).unsqueeze(0)
    full = torch.cat([ctx_embeds, z_cast, tail_embeds], dim=0).unsqueeze(0)

    out = base_model(inputs_embeds=full)
    logits = out.logits[0]  # (L, V)

    ctx_len = ctx_embeds.size(0)
    tail_len = tail_embeds.size(0)
    pred_logits = logits[ctx_len : ctx_len + tail_len]  # (tail_len, V)
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    selected = log_probs.gather(-1, tail_ids.unsqueeze(-1)).squeeze(-1)  # (tail_len,)
    return selected.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan_24_dpo.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Distributed setup ────────────────────────────────────────────────────
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if distributed:
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    # ── Load frozen base in 4-bit ────────────────────────────────────────────
    model_name = config["model"]["base_model"]
    if rank == 0:
        print(f"Loading frozen base: {model_name} (world={world_size})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": device},
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    hidden_dim = base_model.config.hidden_size
    if rank == 0:
        print(f"Hidden dim: {hidden_dim}", flush=True)

    if distributed:
        torch.distributed.barrier()

    # ── ProjMLP ──────────────────────────────────────────────────────────────
    target_norm = config["model"].get("plan_vector_scale")
    proj = ProjMLP(
        hidden_dim,
        config["model"]["proj_hidden_dims"],
        target_norm=target_norm,
    ).to(device).float()

    if distributed:
        proj = DDP(
            proj, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, broadcast_buffers=False,
        )
        trainable_params = list(proj.module.parameters())
    else:
        trainable_params = list(proj.parameters())

    if rank == 0:
        total = sum(p.numel() for p in trainable_params)
        print(f"ProjMLP trainable params: {total:,} (target_norm={target_norm})",
              flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = Game24DPODataset(
        tokenizer,
        config["data"]["train_data"],
        config["data"]["refs_cache"],
        max_ctx_len=config["data"]["max_ctx_len"],
        max_tail_len=config["data"]["max_tail_len"],
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    epochs = config["training"]["epochs"]
    grad_accum = config["training"]["grad_accum"]
    total_steps = (len(dataloader) * epochs) // grad_accum
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    optimizer = AdamW(trainable_params, lr=config["training"]["lr"], weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    beta = config["training"]["beta"]
    lambda_z_l2 = config["training"].get("lambda_z_l2", 0.0)
    lambda_dpo = config["training"].get("lambda_dpo", 1.0)    # weight on DPO term
    lambda_plan = config["training"].get("lambda_plan", 0.0)   # weight on CE-on-positive (plan loss)

    # ── Output dirs ──────────────────────────────────────────────────────────
    output_dir = config["training"]["output_dir"]
    log_dir = config["training"]["log_dir"]
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "plan_24_dpo_train.jsonl")

    if rank == 0:
        print(f"Training: {len(dataset)} pairs, {epochs} epochs, "
              f"grad_accum={grad_accum}, total_steps={total_steps} (per rank), "
              f"β={beta}", flush=True)

    embed_table = base_model.get_input_embeddings()
    global_step = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        running = {
            "loss": 0.0, "margin": 0.0, "margin_pos_frac": 0.0,
            "z_norm": 0.0, "pairs_seen": 0,
        }
        sample_losses = []
        sample_margins = []
        sample_z_norms = []

        for batch_idx, batch in enumerate(dataloader):
            ctx_ids = batch["ctx_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            lp_ref_pos = batch["log_pi_ref_pos"]
            lp_ref_neg = batch["log_pi_ref_neg"]

            # Forward frozen base on ctx to get hidden at boundary (last ctx token)
            with torch.no_grad():
                ctx_out = base_model(
                    input_ids=ctx_ids.unsqueeze(0),
                    output_hidden_states=True,
                )
                h_boundary = ctx_out.hidden_states[-1][0, -1]  # (H,)

            # Compute z (grad-enabled)
            h_in = h_boundary.unsqueeze(0).float()
            _, z = proj(h_in)
            z = z.squeeze(0)  # (H,)

            # Forward with z for positive and negative
            logp_pos_with = seq_logprob_with_z(
                base_model, embed_table, ctx_ids, z, pos_ids, device,
            )
            logp_neg_with = seq_logprob_with_z(
                base_model, embed_table, ctx_ids, z, neg_ids, device,
            )

            pos_ratio = logp_pos_with - lp_ref_pos
            neg_ratio = logp_neg_with - lp_ref_neg
            margin_val = beta * (pos_ratio - neg_ratio)
            dpo_loss = -F.logsigmoid(margin_val)

            # CE-on-positive (plan loss): mean per-token neg log-prob over r+ tokens
            pos_len = pos_ids.size(0)
            plan_loss = -logp_pos_with / max(pos_len, 1)

            loss = lambda_plan * plan_loss + lambda_dpo * dpo_loss
            if lambda_z_l2 > 0:
                z_reg = lambda_z_l2 * (z.float() ** 2).sum()
                loss = loss + z_reg

            sample_losses.append(loss)
            sample_margins.append(margin_val.detach().item())
            sample_z_norms.append(z.detach().float().norm().item())
            if batch_idx == 0 and rank == 0 and global_step == 0:
                print(f"  init | plan={plan_loss.item():.4f} dpo={dpo_loss.item():.4f} "
                      f"λ_plan={lambda_plan} λ_dpo={lambda_dpo}", flush=True)

            # Every grad_accum steps: one backward per batch iter (DDP-compatible)
            if (batch_idx + 1) % grad_accum == 0:
                if sample_losses:
                    avg_loss = sum(sample_losses) / len(sample_losses)
                    (avg_loss / grad_accum).backward()
                else:
                    # Dummy backward for DDP sync (shouldn't happen, but safe)
                    dummy = torch.zeros(1, hidden_dim, device=device, dtype=torch.float32)
                    _, z_dummy = (proj.module if distributed else proj)(dummy)
                    (z_dummy.sum() * 0.0).backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Track
                mean_loss = float(sum(l.item() for l in sample_losses) / len(sample_losses))
                mean_margin = sum(sample_margins) / len(sample_margins)
                pos_frac = sum(1 for m in sample_margins if m > 0) / len(sample_margins)
                mean_zn = sum(sample_z_norms) / len(sample_z_norms)
                min_zn = min(sample_z_norms)
                max_zn = max(sample_z_norms)

                running["loss"] += mean_loss
                running["margin"] += mean_margin
                running["margin_pos_frac"] += pos_frac
                running["z_norm"] += mean_zn
                running["pairs_seen"] += 1

                if rank == 0 and global_step % 5 == 0:
                    print(
                        f"  step {global_step}/{total_steps} | "
                        f"loss={mean_loss:.4f} | margin={mean_margin:+.3f} | "
                        f"pos_frac={pos_frac:.3f} | ‖z‖={mean_zn:.3f} "
                        f"[{min_zn:.2f}-{max_zn:.2f}] | "
                        f"lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                    with open(log_file, "a") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch,
                            "loss": round(mean_loss, 4),
                            "margin": round(mean_margin, 4),
                            "margin_pos_frac": round(pos_frac, 4),
                            "z_norm_mean": round(mean_zn, 4),
                            "z_norm_min": round(min_zn, 4),
                            "z_norm_max": round(max_zn, 4),
                            "lr": scheduler.get_last_lr()[0],
                        }) + "\n")

                    # Sanity warnings
                    if mean_zn > 100 or (not math.isfinite(mean_zn)):
                        print(f"  ⚠️  ‖z‖ exploding: {mean_zn}", flush=True)
                    if mean_zn < 0.01:
                        print(f"  ⚠️  ‖z‖ collapsing: {mean_zn}", flush=True)

                sample_losses = []
                sample_margins = []
                sample_z_norms = []

        if rank == 0 and running["pairs_seen"] > 0:
            n = running["pairs_seen"]
            print(
                f"Epoch {epoch}: loss={running['loss']/n:.4f} | "
                f"margin={running['margin']/n:+.3f} | "
                f"pos_frac={running['margin_pos_frac']/n:.3f} | "
                f"‖z‖={running['z_norm']/n:.3f}",
                flush=True,
            )

    # ── Save ProjMLP (rank 0) ────────────────────────────────────────────────
    if rank == 0:
        print(f"Saving ProjMLP to {output_dir}", flush=True)
        state = proj.module.state_dict() if distributed else proj.state_dict()
        torch.save(state, os.path.join(output_dir, "proj.pt"))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        print("Done.", flush=True)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
