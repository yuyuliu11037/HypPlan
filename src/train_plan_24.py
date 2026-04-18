"""Planning vector training for Game of 24.

Freezes the SFT-trained LLM and trains only the ProjMLP.
For each step boundary, computes a planning vector z from the hidden state
and inserts it as a virtual token before the step tokens.
Loss: cross-entropy on step tokens only.

Supports DDP: each GPU loads the frozen 4-bit base independently; only
ProjMLP gradients sync across ranks.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from src.projections import ProjMLP
from src.dataset_24_plan import Game24PlanDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan_24.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Distributed setup
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

    # Load frozen base model (merged SFT) in 4-bit on THIS rank's device
    model_name = config["model"]["base_model"]
    if rank == 0:
        print(f"Loading frozen base: {model_name} (world_size={world_size})")

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
    for param in base_model.parameters():
        param.requires_grad = False

    hidden_dim = base_model.config.hidden_size
    if rank == 0:
        print(f"Hidden dim: {hidden_dim}", flush=True)

    # Sync all ranks before DDP construction (4-bit model loading times vary)
    if distributed:
        torch.distributed.barrier()

    # ProjMLP (trainable, float32 for stable gradients)
    proj = ProjMLP(
        hidden_dim,
        config["model"]["proj_hidden_dims"],
        target_norm=config["model"].get("plan_vector_scale", 1.0),
    ).to(device).float()

    if distributed:
        proj = DDP(proj, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=False, broadcast_buffers=False)
        trainable_params = list(proj.module.parameters())
    else:
        trainable_params = list(proj.parameters())

    if rank == 0:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"ProjMLP trainable params: {total_params:,}")

    # Dataset
    dataset = Game24PlanDataset(
        tokenizer,
        config["data"]["train_data"],
        max_seq_len=config["data"]["max_seq_len"],
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
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer & scheduler
    epochs = config["training"]["epochs"]
    grad_accum = config["training"]["grad_accum"]
    total_steps = (len(dataloader) * epochs) // grad_accum
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    optimizer = AdamW(trainable_params, lr=config["training"]["lr"], weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Output dirs
    output_dir = config["training"]["output_dir"]
    log_dir = config["training"]["log_dir"]
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "plan_24_train.jsonl")

    if rank == 0:
        print(f"Training: {len(dataset)} samples, {epochs} epochs, "
              f"grad_accum={grad_accum}, total_steps={total_steps} (per-rank)")

    embed_table = base_model.get_input_embeddings()
    global_step = 0

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        z_norms = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_positions = batch["boundary_positions"].to(device)

            # Pass 1: get hidden states from full sequence (no grad)
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]  # (B, L, H)

            B = input_ids.size(0)
            batch_loss = 0.0
            batch_steps = 0

            # Accumulate all step losses into one scalar, then a single backward
            # (required for DDP: each rank must have the same number of .backward() calls)
            sample_losses = []

            for b in range(B):
                valid_bp = boundary_positions[b][boundary_positions[b] >= 0]
                K = valid_bp.size(0)
                real_len = attention_mask[b].sum().item()

                for i in range(K):
                    bpos = valid_bp[i].item()

                    # Step token range
                    step_start = bpos + 1
                    step_end = valid_bp[i + 1].item() + 1 if i < K - 1 else real_len
                    if step_start >= step_end:
                        continue
                    step_ids = input_ids[b, step_start:step_end]

                    # Compute planning vector (grad-enabled, float32)
                    h_i = hidden_states[b, bpos].unsqueeze(0).float()  # (1, H)
                    _, z_i = proj(h_i)  # (1, H) float32
                    z_norms.append(z_i.detach().float().norm().item())

                    # Build [prefix, z_i, step] as embeddings
                    with torch.no_grad():
                        prefix_embeds = embed_table(input_ids[b, :bpos + 1])
                        step_embeds = embed_table(step_ids)
                    embed_dtype = prefix_embeds.dtype

                    prefix_len = prefix_embeds.size(0)
                    s_len = step_embeds.size(0)

                    full_embeds = torch.cat(
                        [prefix_embeds, z_i.to(embed_dtype), step_embeds], dim=0
                    ).unsqueeze(0)  # (1, L_i, H)
                    L_i = full_embeds.size(1)

                    # Labels: predict step tokens after z_i
                    full_labels = torch.full(
                        (1, L_i), -100, dtype=torch.long, device=device
                    )
                    full_labels[0, prefix_len + 1: prefix_len + 1 + s_len] = step_ids

                    # Forward through frozen base
                    out = base_model(inputs_embeds=full_embeds)
                    logits = out.logits

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = full_labels[:, 1:].contiguous()
                    loss_i = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                    sample_losses.append(loss_i)
                    batch_loss += loss_i.item()
                    batch_steps += 1

            # Single backward per batch iter (DDP-compatible: same sync count per rank)
            if sample_losses:
                avg_loss = sum(sample_losses) / len(sample_losses)
                (avg_loss / grad_accum).backward()
            else:
                # No valid boundaries in this batch — still participate in DDP sync
                # by running a dummy forward+backward through proj (zero loss)
                dummy_h = torch.zeros(1, hidden_dim, device=device, dtype=torch.float32)
                _, dummy_z = proj(dummy_h)
                (dummy_z.sum() * 0.0).backward()

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_batch_loss = batch_loss / max(batch_steps, 1)
                epoch_loss += avg_batch_loss
                epoch_steps += 1

                if rank == 0 and global_step % 10 == 0:
                    z_norm_mean = sum(z_norms[-batch_steps:]) / max(len(z_norms[-batch_steps:]), 1)
                    print(f"  step {global_step}/{total_steps} | "
                          f"loss={avg_batch_loss:.4f} | "
                          f"z_norm={z_norm_mean:.4f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                    with open(log_file, "a") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "epoch": epoch,
                            "loss": round(avg_batch_loss, 4),
                            "z_norm": round(z_norm_mean, 4),
                            "lr": scheduler.get_last_lr()[0],
                        }) + "\n")

                batch_loss = 0.0
                batch_steps = 0

        if rank == 0:
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}", flush=True)

    # Save ProjMLP (rank 0 only)
    if rank == 0:
        print(f"Saving ProjMLP to {output_dir}")
        state = proj.module.state_dict() if distributed else proj.state_dict()
        torch.save(state, os.path.join(output_dir, "proj.pt"))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        print("Done.")

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
