"""Stage 2: Structurize Proj with tree loss."""
from __future__ import annotations

import argparse
import os
import random

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset_stage2 import Stage2Dataset, collate_stage2
from src.model.hyperbolic import lorentz_distance
from src.model.plan_model import HypPlanModel


def compute_tree_loss(all_t: list[torch.Tensor], all_node_ids: list[list[int]],
                      node_distances: dict, c: torch.Tensor,
                      num_pairs: int = 256) -> torch.Tensor:
    """Compute L_tree: MSE between hyperbolic distances and scaled tree distances.

    Args:
        all_t: List of (num_steps_i, hyp_dim+1) tensors, one per generation.
        all_node_ids: List of node_id lists, one per generation.
        node_distances: Dict mapping (node_a, node_b) -> tree distance.
        c: Learnable scaling factor.
        num_pairs: Number of pairs to sample.
    Returns:
        Scalar tree loss.
    """
    # Flatten all steps across generations
    flat_t = []
    flat_nids = []
    for t_gen, nids_gen in zip(all_t, all_node_ids):
        for step_idx in range(t_gen.size(0)):
            flat_t.append(t_gen[step_idx])
            flat_nids.append(nids_gen[step_idx])

    if len(flat_t) < 2:
        return torch.tensor(0.0, device=flat_t[0].device if flat_t else "cpu")

    flat_t = torch.stack(flat_t)  # (N, hyp_dim+1)
    N = flat_t.size(0)

    # Sample pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    if len(all_pairs) > num_pairs:
        pairs = random.sample(all_pairs, num_pairs)
    else:
        pairs = all_pairs

    if not pairs:
        return torch.tensor(0.0, device=flat_t.device)

    idx_a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    idx_b = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    t_a = flat_t[idx_a]
    t_b = flat_t[idx_b]

    d_hyp = lorentz_distance(t_a, t_b)

    # Look up tree distances
    d_tree_vals = []
    for ia, ib in pairs:
        nid_a = flat_nids[ia]
        nid_b = flat_nids[ib]
        d = node_distances.get((nid_a, nid_b), node_distances.get((nid_b, nid_a), 0))
        d_tree_vals.append(float(d))
    d_tree = torch.tensor(d_tree_vals, device=flat_t.device, dtype=flat_t.dtype)

    loss = ((d_hyp - c * d_tree) ** 2).mean()
    return loss


def train_stage2(config_path: str, local_rank: int = -1):
    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    # Build model
    model = HypPlanModel(config)
    model.freeze_base_model()
    model.base_model.gradient_checkpointing_enable()
    model.to(torch.bfloat16).to(device)

    # Load Stage 1 checkpoint
    stage1_ckpt_path = os.path.join(config["stage1"]["output_dir"], "checkpoint.pt")
    if os.path.exists(stage1_ckpt_path):
        ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=True)
        model.proj.load_state_dict(ckpt["proj"])
        model.project_back.load_state_dict(ckpt["project_back"])
        if is_main:
            print(f"Loaded Stage 1 checkpoint from {stage1_ckpt_path}")
    else:
        if is_main:
            print("WARNING: No Stage 1 checkpoint found, starting Proj from scratch")

    # Learnable scaling factor c
    c = torch.nn.Parameter(
        torch.tensor(config["stage2"]["c_init"], device=device, dtype=torch.float32)
    )

    if distributed:
        model.proj = torch.nn.parallel.DistributedDataParallel(
            model.proj, device_ids=[local_rank]
        )
        model.project_back = torch.nn.parallel.DistributedDataParallel(
            model.project_back, device_ids=[local_rank]
        )

    # Dataset
    dataset = Stage2Dataset(
        filtered_path=config["data"]["math_filtered"],
        trees_path=config["data"]["reasoning_trees"],
        tokenizer=model.tokenizer,
        max_seq_len=config["data"]["max_seq_len"],
        step_delimiter=config["data"]["step_delimiter"],
    )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=1,  # one problem at a time
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_stage2,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer
    trainable_params = list(model.proj.parameters()) + list(model.project_back.parameters()) + [c]
    optimizer = AdamW(trainable_params, lr=config["stage2"]["lr"], weight_decay=0.01)

    grad_accum = config["stage2"]["grad_accum"]
    epochs = config["stage2"]["epochs"]
    total_steps = (len(loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * config["stage2"].get("warmup_ratio", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    tree_loss_weight = config["stage2"]["tree_loss_weight"]
    num_pairs = config["stage2"].get("num_pairs", 256)

    if is_main:
        print(f"Stage 2: {len(dataset)} problems, {len(loader)} batches/epoch")
        print(f"Tree loss weight: {tree_loss_weight}, num_pairs: {num_pairs}")
        print(f"Total steps: {total_steps}")

    model.base_model.eval()
    model.proj.train()
    model.project_back.train()

    global_step = 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_plan_loss = 0.0
        epoch_tree_loss = 0.0
        num_batches = 0

        for batch_idx, batch_list in enumerate(loader):
            problem_data = batch_list[0]  # single problem
            gens = problem_data["generations"]
            node_distances = problem_data["node_distances"]

            # Process each generation: backward plan loss per-gen to free activations
            total_plan_loss_val = 0.0
            all_t = []
            all_node_ids = []
            valid_gen_count = 0

            for gen_data in gens:
                input_ids = gen_data["input_ids"].unsqueeze(0).to(device)
                attention_mask = gen_data["attention_mask"].unsqueeze(0).to(device)
                labels = gen_data["labels"].unsqueeze(0).to(device)
                boundary_pos = gen_data["boundary_positions"].unsqueeze(0).to(device)
                inject_pos = gen_data["inject_positions"].unsqueeze(0).to(device)

                if boundary_pos.size(1) == 0:
                    continue

                loss, t = model.forward_stage2(
                    input_ids, attention_mask, labels, boundary_pos, inject_pos
                )

                # Backward plan loss immediately to free LLM activations
                (loss / (len(gens) * grad_accum)).backward()
                total_plan_loss_val += loss.item()
                valid_gen_count += 1

                # Detach planning vectors for tree loss (no LLM grad needed)
                valid_mask = boundary_pos[0] >= 0
                t_valid = t[0][valid_mask].detach().requires_grad_(True)
                node_ids = gen_data["node_ids"].tolist()
                n = min(t_valid.size(0), len(node_ids))
                all_t.append(t_valid[:n])
                all_node_ids.append(node_ids[:n])

            if valid_gen_count == 0:
                continue

            plan_loss_val = total_plan_loss_val / valid_gen_count

            # Tree loss — only backprops through Proj (detached from LLM)
            tree_loss = compute_tree_loss(
                all_t, all_node_ids, node_distances, c, num_pairs
            )

            (tree_loss_weight * tree_loss / grad_accum).backward()

            epoch_plan_loss += plan_loss_val
            epoch_tree_loss += tree_loss.item()
            num_batches += 1

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 50 == 0:
                    print(
                        f"  Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"L_plan: {plan_loss_val:.4f} | L_tree: {tree_loss.item():.4f} | "
                        f"c: {c.item():.4f}"
                    )

        avg_plan = epoch_plan_loss / max(num_batches, 1)
        avg_tree = epoch_tree_loss / max(num_batches, 1)
        if is_main:
            print(
                f"Epoch {epoch+1}/{epochs} — L_plan: {avg_plan:.4f}, "
                f"L_tree: {avg_tree:.4f}, c: {c.item():.4f}"
            )

    # Save checkpoint
    if is_main:
        output_dir = config["stage2"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        proj_state = model.proj.module.state_dict() if distributed else model.proj.state_dict()
        pb_state = model.project_back.module.state_dict() if distributed else model.project_back.state_dict()
        torch.save({
            "proj": proj_state,
            "project_back": pb_state,
            "c": c.data,
            "optimizer": optimizer.state_dict(),
        }, os.path.join(output_dir, "checkpoint.pt"))
        print(f"Saved Stage 2 checkpoint to {output_dir}")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train_stage2(args.config, args.local_rank)
