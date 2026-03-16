from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


class BaselineSFTDataset(Dataset):
    """Plain SFT dataset: problem as context, solution steps as targets (no [PLAN] tokens)."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int,
        limit: int | None = None,
    ) -> None:
        with open(data_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            records = records[:limit]

        self.samples: list[dict] = []
        for rec in records:
            if "steps" not in rec or len(rec["steps"]) <= 1:
                continue
            step_texts = [str(s["step_text"]).strip() for s in rec["steps"]]
            question_ids = tokenizer.encode(rec["problem"] + "\n\n", add_special_tokens=False)
            solution_ids = tokenizer.encode(" ".join(step_texts), add_special_tokens=False)
            if not solution_ids:
                continue
            eos = tokenizer.eos_token_id
            input_ids = question_ids + solution_ids + [eos]
            if len(input_ids) > max_seq_len:
                continue
            # Mask question tokens; train on solution + eos
            labels = [-100] * len(question_ids) + solution_ids + [eos]
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq: list[int], fill: int) -> list[int]:
        return seq + [fill] * (max_len - len(seq))

    return {
        "input_ids": torch.tensor([pad(x["input_ids"], 0) for x in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch], dtype=torch.long),
        "labels": torch.tensor([pad(x["labels"], -100) for x in batch], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--limit", type=int, default=8000)
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
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--deepspeed", default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_setup()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    train_set = BaselineSFTDataset(
        args.data_path, tokenizer, args.max_seq_len, limit=args.limit
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loop = tqdm(
            train_loader,
            disable=rank != 0,
            desc=f"baseline sft epoch {epoch + 1}/{args.num_epochs}",
        )
        optimizer.zero_grad(set_to_none=True)
        for batch in loop:
            batch = move_batch(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            (outputs.loss / args.gradient_accumulation_steps).backward()
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loop.set_postfix(loss=float(outputs.loss.detach().cpu()))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(args.output_dir, "lora_adapters"))
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
