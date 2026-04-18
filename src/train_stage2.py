"""Stage-2 trainer: LoRA + UpProjector on top of frozen SFT + frozen head.

At each step boundary:
  state_ids (canonical text) -> frozen base (LoRA disabled) -> hidden ->
  frozen head -> z_hyp -> trainable UpProjector -> z_inj -> virtual token
  before the step's tokens. CE loss on step tokens.

Forks src/train_plan_24.py; reuses its injection mechanics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from src.head import HyperbolicHead, UpProjector
from src.dataset_24_stage2 import Game24Stage2Dataset, collate_fn


def setup_distributed():
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
    return distributed, rank, world_size, local_rank, device


def load_head(config: dict, device, hidden_dim: int) -> HyperbolicHead:
    ckpt = torch.load(config["model"]["head_checkpoint"], map_location="cpu",
                      weights_only=False)
    head_cfg = ckpt["config"]["model"]
    head = HyperbolicHead(
        in_dim=ckpt["in_dim"],
        hyp_dim=head_cfg["hyp_dim"],
        hidden_dims=head_cfg["head_hidden_dims"],
        manifold=head_cfg["manifold"],
    ).to(device).to(torch.float32)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--random_z", action="store_true",
                        help="Null baseline: inject Gaussian unit-norm noise in place of "
                             "up_proj(head(state)). Trains LoRA + up_proj on pure noise so "
                             "the LoRA learns whatever it can from CE alone.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    random_z_train = args.random_z

    distributed, rank, world_size, local_rank, device = setup_distributed()

    # Deterministic seed so every rank produces IDENTICAL LoRA + UpProjector
    # init weights. Without this, each rank would randomly init the trainable
    # params and diverge on step 1.
    torch.manual_seed(1234)
    import numpy as _np; _np.random.seed(1234)

    # --- Base model (frozen, 4-bit) + LoRA (trainable) ---
    model_name = config["model"]["base_model"]
    if rank == 0:
        print(f"Loading base: {model_name} (world_size={world_size})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_cfg, trust_remote_code=True,
        device_map={"": device},
    )
    for p in base_model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"].get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()
    if rank == 0:
        model.print_trainable_parameters()

    hidden_dim = base_model.config.hidden_size

    # --- Head (frozen) + UpProjector (trainable) ---
    head = load_head(config, device, hidden_dim)
    up_proj = UpProjector(
        in_dim=config["model"]["hyp_dim"]
                + (1 if config["model"]["manifold"] == "lorentz" else 0),
        hidden=config["model"]["up_proj_hidden"],
        out_dim=hidden_dim,
    ).to(device).float()

    # Deliberately NOT wrapping in DDP — the training step has a variable-K
    # per-boundary loop plus disable_adapter() sub-forwards that make bucket
    # ordering diverge across ranks and deadlock the auto-reducer. Instead we
    # broadcast params once and manually all-reduce gradients after backward.
    up_proj_ref = up_proj
    model_module = model

    lora_params = [p for p in model.parameters() if p.requires_grad]
    up_params = list(up_proj_ref.parameters())
    trainable = lora_params + up_params
    if rank == 0:
        n_lora = sum(p.numel() for p in lora_params)
        n_up = sum(p.numel() for p in up_params)
        print(f"Trainable: LoRA={n_lora:,}  UpProjector={n_up:,}", flush=True)

    # Deterministic seed above guarantees identical init across ranks — no
    # broadcast collective needed (which previously deadlocked because ranks
    # entered the collective loop at skewed times after 4-bit model loading).
    if distributed:
        dist.barrier()   # ensure all ranks finished loading before first step

    # --- Data ---
    ds = Game24Stage2Dataset(tokenizer, config["data"]["train_data"],
                              max_seq_len=config["data"]["max_seq_len"])
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=True) if distributed else None
    loader = DataLoader(
        ds, batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    # --- Optimizer ---
    epochs = int(config["training"]["epochs"])
    grad_accum = int(config["training"]["grad_accum"])
    total_steps = (len(loader) * epochs) // grad_accum
    warmup = int(total_steps * float(config["training"].get("warmup_ratio", 0.05)))
    lr = float(config["training"]["lr"])

    optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir = Path(config["training"]["output_dir"])
    log_dir = Path(config["training"]["log_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
    log_file = log_dir / "stage2_train.jsonl"

    embed_table = model_module.get_input_embeddings()

    global_step = 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_count = 0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_positions = batch["boundary_positions"].to(device)
            state_valid = batch["state_valid"].to(device)
            state_ids_list = [s.to(device) for s in batch["state_input_ids"]]
            state_mask_list = [s.to(device) for s in batch["state_attention_mask"]]

            B = input_ids.size(0)

            # 1. Compute z_inj for every (sample, boundary). With random_z_train,
            #    skip head/up_proj entirely and use Gaussian unit-norm noise so
            #    the LoRA gets the same noise distribution it will see at
            #    inference (--random_z in generate_24_stage2.py).
            K = len(state_ids_list)
            z_inj_per_k: list = []   # list of (B, hidden_dim) tensors
            for k in range(K):
                if random_z_train:
                    g = torch.randn(B, hidden_dim, device=device, dtype=torch.float32)
                    z_inj = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    z_inj_per_k.append(z_inj)
                    continue
                ids_k = state_ids_list[k]       # (B, S_k)
                mask_k = state_mask_list[k]
                with torch.no_grad():
                    with model_module.disable_adapter():
                        out = model_module(
                            input_ids=ids_k, attention_mask=mask_k,
                            output_hidden_states=True,
                        )
                        last_h = out.hidden_states[-1]      # (B, S_k, H)
                        last_idx = mask_k.sum(dim=1) - 1   # (B,)
                        h_row = last_h[torch.arange(B, device=device), last_idx]
                    z_hyp = head(h_row.float())            # (B, hyp_out)
                z_inj = up_proj_ref(z_hyp)                 # (B, hidden_dim)  trainable
                z_inj_per_k.append(z_inj)

            # 2. Per-sample per-boundary: embed prefix + z + step_tokens, CE loss
            sample_losses: list = []
            real_lens = attention_mask.sum(dim=1).long()

            for b in range(B):
                bp_row = boundary_positions[b]
                valid_mask = bp_row >= 0
                valid_bp = bp_row[valid_mask]
                K_b = int(valid_bp.size(0))
                for i in range(K_b):
                    bpos = int(valid_bp[i].item())
                    step_start = bpos + 1
                    step_end = int(valid_bp[i + 1].item() + 1) if i < K_b - 1 \
                               else int(real_lens[b].item())
                    if step_start >= step_end:
                        continue
                    step_ids = input_ids[b, step_start:step_end]

                    z_i = z_inj_per_k[i][b]  # (hidden_dim,)

                    with torch.no_grad():
                        prefix_embeds = embed_table(input_ids[b, :bpos + 1])
                        step_embeds = embed_table(step_ids)
                    embed_dtype = prefix_embeds.dtype

                    full_embeds = torch.cat([
                        prefix_embeds,
                        z_i.to(embed_dtype).unsqueeze(0),
                        step_embeds,
                    ], dim=0).unsqueeze(0)
                    L_i = full_embeds.size(1)

                    full_labels = torch.full(
                        (1, L_i), -100, dtype=torch.long, device=device,
                    )
                    p_len = prefix_embeds.size(0)
                    s_len = step_embeds.size(0)
                    full_labels[0, p_len + 1: p_len + 1 + s_len] = step_ids

                    out = model(inputs_embeds=full_embeds)
                    logits = out.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = full_labels[:, 1:].contiguous()
                    loss_i = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1), ignore_index=-100,
                    )
                    sample_losses.append(loss_i)

            if sample_losses:
                loss = sum(sample_losses) / len(sample_losses)
            else:
                # DDP sync: dummy grad through up_proj
                dummy = torch.zeros(1, config["model"]["hyp_dim"]
                                    + (1 if config["model"]["manifold"] == "lorentz" else 0),
                                    device=device)
                loss = (up_proj_ref(dummy).sum() * 0.0)

            (loss / grad_accum).backward()
            epoch_loss += float(loss.item())
            epoch_count += 1

            if (batch_idx + 1) % grad_accum == 0:
                # Manual gradient averaging across ranks. Skipped in single-GPU.
                if distributed:
                    for p in trainable:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p.data)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if rank == 0 and (global_step % 10 == 0 or global_step == 1):
                    avg = epoch_loss / max(epoch_count, 1)
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                          flush=True)
                    with open(log_file, "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "epoch": epoch,
                            "loss": round(avg, 4),
                            "lr": scheduler.get_last_lr()[0],
                        }) + "\n")

        if rank == 0:
            print(f"== epoch {epoch} avg_loss={epoch_loss/max(epoch_count,1):.4f} ==",
                  flush=True)

    if rank == 0:
        model_module.save_pretrained(out_dir / "lora")
        torch.save(up_proj_ref.state_dict(), out_dir / "up_projector.pt")
        print(f"Saved stage-2 artifacts to {out_dir}", flush=True)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
