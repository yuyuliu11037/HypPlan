from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
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
    parser.add_argument("--proj_checkpoint", required=True)
    parser.add_argument("--proj_type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--presplit_data", action="store_true")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--deepspeed", default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_setup()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PLAN]"]})
    plan_token_id = tokenizer.convert_tokens_to_ids("[PLAN]")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    base.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_config)
    base.to(device)

    proj = ProjectionModule(base.config.hidden_size, args.proj_type).to(device)
    proj.load_state_dict(torch.load(args.proj_checkpoint, map_location=device))
    plan_token_delta = torch.nn.Parameter(
        torch.zeros(base.config.hidden_size, device=device, dtype=next(base.parameters()).dtype)
    )

    model = PlanningQwen(
        base_model=base,
        proj=proj,
        plan_token_id=plan_token_id,
        structural_loss="simple",
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

    lora_params = [p for p in base.parameters() if p.requires_grad]
    proj_params = [p for p in proj.parameters() if p.requires_grad]
    trainable_params = lora_params + proj_params + [plan_token_delta]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loop = tqdm(train_loader, disable=rank != 0, desc=f"stage2 epoch {epoch+1}/{args.num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in loop:
            batch = move_batch(batch, device)
            outputs = model.module.stage2_forward(  # type: ignore[union-attr]
                batch=batch,
                plan_token_delta=plan_token_delta,
            ) if hasattr(model, "module") else model.stage2_forward(
                batch=batch,
                plan_token_delta=plan_token_delta,
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
        base_to_save = model_to_save.base_model
        base_to_save.save_pretrained(os.path.join(args.output_dir, "lora_adapters"))
        torch.save(model_to_save.proj.state_dict(), os.path.join(args.output_dir, "proj.pt"))
        torch.save(plan_token_delta.detach().cpu(), os.path.join(args.output_dir, "plan_token_delta.pt"))
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

