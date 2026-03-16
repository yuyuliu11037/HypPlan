from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import PlanningTokenDataset, collate_fn
from src.model.planning_model import PlanningQwen
from src.model.proj import ProjectionModule


def ddp_setup() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--proj_type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--structural_loss", default="simple", choices=["simple", "contrastive"])
    parser.add_argument("--lambda_seg", type=float, default=0.1)
    parser.add_argument("--lambda_depth", type=float, default=0.1)
    parser.add_argument(
        "--reconstruct_mode",
        type=str,
        default="contextual",
        choices=["contextual", "isolated"],
        help=(
            "How to compute reconstruction loss. "
            "'contextual': two-pass over full sequence with t_i injected at embedding level. "
            "'isolated': per-step forward passes with only [PLAN]+t_i as prefix."
        ),
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--presplit_data", action="store_true")
    parser.add_argument(
        "--max_step_len",
        type=int,
        default=256,
        help="Maximum number of tokens per reasoning step in isolated reconstruction mode.",
    )
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--deepspeed", default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_setup()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PLAN]"]})
    plan_token_id = tokenizer.convert_tokens_to_ids("[PLAN]")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.requires_grad_(False)
    base_model.to(device)

    proj = ProjectionModule(base_model.config.hidden_size, args.proj_type).to(device)
    model = PlanningQwen(
        base_model=base_model,
        proj=proj,
        plan_token_id=plan_token_id,
        structural_loss=args.structural_loss,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_set = PlanningTokenDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        split="train",
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        limit=args.limit,
        presplit=args.presplit_data,
    )
    sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    trainable_params = list(model.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loop = tqdm(train_loader, disable=rank != 0, desc=f"stage1 epoch {epoch+1}/{args.num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in loop:
            batch = move_batch(batch, device)
            stage1_model = model.module if hasattr(model, "module") else model  # type: ignore[union-attr]
            outputs = stage1_model.stage1_forward(
                batch=batch,
                reconstruct_mode=args.reconstruct_mode,
                lambda_seg=args.lambda_seg,
                lambda_depth=args.lambda_depth,
                max_step_len=args.max_step_len,
            )

            (outputs["loss"] / args.gradient_accumulation_steps).backward()
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loop.set_postfix(loss=float(outputs["loss"].detach().cpu()))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.proj.state_dict(), os.path.join(args.output_dir, "proj.pt"))
        torch.save(
            {
                "segment_head": model_to_save.segment_head.state_dict(),
                "depth_head": model_to_save.depth_head.state_dict(),
            },
            os.path.join(args.output_dir, "structural_heads.pt"),
        )
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

