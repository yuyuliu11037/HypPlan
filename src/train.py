from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from src.data.dataset import PlanningTokenDataset, collate_fn
from src.losses.contrastive_structural import compute_contrastive_structural_losses
from src.losses.simple_structural import compute_simple_structural_losses
from src.model.planning_model import PlanningQwen

try:
    import deepspeed
except Exception:
    deepspeed = None


@dataclass
class TrainMetrics:
    lm_loss: float
    seg_loss: float
    depth_loss: float
    total_loss: float


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return rank, world_size
    return 0, 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_scalar(value: torch.Tensor, world_size: int) -> float:
    v = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v = v / world_size
    return float(v.item())


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    world_size: int,
    rank: int,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def compute_structural_losses(
    model: PlanningQwen,
    batch: dict[str, torch.Tensor],
    plan_vectors: torch.Tensor,
    structural_loss: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = batch["plan_mask"]
    flat_plan_vectors = plan_vectors[valid]
    flat_seg_raw = batch["segment_ids_raw"][valid]
    flat_seg_labels = batch["segment_labels"][valid]
    flat_depth_targets = batch["depth_labels"][valid]
    flat_solution_ids = batch["solution_ids"][valid]

    if structural_loss == "simple":
        if flat_plan_vectors.numel() == 0:
            z = plan_vectors.new_zeros(())
            return z, z
        seg_logits = model.segment_classifier(flat_plan_vectors)
        depth_pred = model.depth_regressor(flat_plan_vectors).squeeze(-1)
        return compute_simple_structural_losses(
            segment_logits=seg_logits,
            segment_targets=flat_seg_labels,
            depth_preds=depth_pred,
            depth_targets=flat_depth_targets,
        )

    return compute_contrastive_structural_losses(
        plan_vectors=flat_plan_vectors,
        segment_ids_raw=flat_seg_raw,
        depth_targets=flat_depth_targets,
        solution_ids=flat_solution_ids,
        depth_readout=model.depth_readout,
        temperature=0.1,
        margin=1.0,
    )


def evaluate(
    model: PlanningQwen,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    args,
) -> TrainMetrics:
    model.eval()
    lm_vals, seg_vals, depth_vals, total_vals = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_loss_mask=batch["token_loss_mask"],
                plan_positions=batch["plan_positions"],
                plan_mask=batch["plan_mask"],
            )
            seg_loss, depth_loss = compute_structural_losses(
                model=model,
                batch=batch,
                plan_vectors=out.plan_vectors,
                structural_loss=args.structural_loss,
            )
            total_loss = out.lm_loss + args.lambda_seg * seg_loss + args.lambda_depth * depth_loss
            lm_vals.append(out.lm_loss)
            seg_vals.append(seg_loss)
            depth_vals.append(depth_loss)
            total_vals.append(total_loss)

    def _avg(vals):
        if not vals:
            return torch.tensor(0.0, device=device)
        return torch.stack(vals).mean()

    lm = reduce_scalar(_avg(lm_vals), world_size)
    seg = reduce_scalar(_avg(seg_vals), world_size)
    dep = reduce_scalar(_avg(depth_vals), world_size)
    tot = reduce_scalar(_avg(total_vals), world_size)
    model.train()
    return TrainMetrics(lm_loss=lm, seg_loss=seg, depth_loss=dep, total_loss=tot)


def train_one_epoch(
    model: PlanningQwen,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    rank: int,
    world_size: int,
    args,
    ds_engine=None,
) -> TrainMetrics:
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(args.current_epoch)

    lm_meter, seg_meter, depth_meter, total_meter = [], [], [], []
    iterator = tqdm(loader, disable=(not is_main_process(rank)))

    for step_idx, batch in enumerate(iterator):
        batch = move_batch_to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_loss_mask=batch["token_loss_mask"],
            plan_positions=batch["plan_positions"],
            plan_mask=batch["plan_mask"],
        )

        seg_loss, depth_loss = compute_structural_losses(
            model=model,
            batch=batch,
            plan_vectors=out.plan_vectors,
            structural_loss=args.structural_loss,
        )
        total_loss = out.lm_loss + args.lambda_seg * seg_loss + args.lambda_depth * depth_loss

        if ds_engine is not None:
            ds_engine.backward(total_loss)
            ds_engine.step()
        else:
            (total_loss / args.gradient_accumulation_steps).backward()
            should_step = (step_idx + 1) % args.gradient_accumulation_steps == 0
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        lm_meter.append(out.lm_loss.detach())
        seg_meter.append(seg_loss.detach())
        depth_meter.append(depth_loss.detach())
        total_meter.append(total_loss.detach())

        if is_main_process(rank) and (step_idx + 1) % args.logging_steps == 0:
            iterator.set_description(
                f"lm={torch.stack(lm_meter[-args.logging_steps:]).mean().item():.4f} "
                f"seg={torch.stack(seg_meter[-args.logging_steps:]).mean().item():.4f} "
                f"dep={torch.stack(depth_meter[-args.logging_steps:]).mean().item():.4f}"
            )

    def _avg(vals):
        if not vals:
            return torch.tensor(0.0, device=device)
        return torch.stack(vals).mean()

    lm = reduce_scalar(_avg(lm_meter), world_size)
    seg = reduce_scalar(_avg(seg_meter), world_size)
    dep = reduce_scalar(_avg(depth_meter), world_size)
    tot = reduce_scalar(_avg(total_meter), world_size)
    return TrainMetrics(lm_loss=lm, seg_loss=seg, depth_loss=dep, total_loss=tot)


def save_checkpoint(model: PlanningQwen, output_dir: str, epoch: int, val_metrics: TrainMetrics) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epoch_{epoch}.pt")
    torch.save(model.get_trainable_state_dict(), path)
    with open(os.path.join(output_dir, "last_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(val_metrics), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--proj_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument(
        "--structural_loss",
        type=str,
        default="simple",
        choices=["simple", "contrastive"],
    )
    parser.add_argument("--lambda_seg", type=float, default=0.1)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default="")
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    setup_seed(args.seed + rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PLAN]"]})

    train_ds = PlanningTokenDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        split="train",
        seed=args.seed,
        max_seq_len=args.max_seq_len,
    )
    val_ds = PlanningTokenDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        split="val",
        seed=args.seed,
        max_seq_len=args.max_seq_len,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
    )

    model = PlanningQwen(
        model_name=args.model_name,
        proj_type=args.proj_type,
        structural_loss=args.structural_loss,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_train_steps = max(
        1,
        (len(train_loader) * args.num_epochs) // max(1, args.gradient_accumulation_steps),
    )
    warmup_steps = int(args.warmup_ratio * total_train_steps)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    ds_engine = None
    if args.deepspeed:
        if deepspeed is None:
            raise RuntimeError("deepspeed package is not installed but --deepspeed was provided.")
        with open(args.deepspeed, "r", encoding="utf-8") as f:
            ds_config = json.load(f)
        ds_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.trainable_parameters(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )

    best_val = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    for epoch in range(1, args.num_epochs + 1):
        args.current_epoch = epoch
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            rank=rank,
            world_size=world_size,
            args=args,
            ds_engine=ds_engine,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            rank=rank,
            world_size=world_size,
            args=args,
        )

        if is_main_process(rank):
            print(
                f"epoch={epoch} "
                f"train_total={train_metrics.total_loss:.4f} "
                f"val_total={val_metrics.total_loss:.4f}"
            )
            save_checkpoint(model=model, output_dir=args.output_dir, epoch=epoch, val_metrics=val_metrics)
            if val_metrics.total_loss < best_val:
                best_val = val_metrics.total_loss
                torch.save(
                    model.get_trainable_state_dict(),
                    os.path.join(args.output_dir, "proj_best.pt"),
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()
