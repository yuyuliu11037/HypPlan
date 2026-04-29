"""Generic PT-SFT trainer for Qwen-14B-Instruct on `{question, answer}` JSONL.

Trains a LoRA on chat-template-formatted (question, answer) pairs with
completion-only loss (only the answer tokens contribute). The `answer`
field is expected to already include `<PLAN:...>` planning-token tags
when running the planning-tokens variant — this trainer is agnostic to
that.

Inputs (config keys, mirroring `configs/sft_pt_*_qwen14b.yaml`):
    model.base_model, model.lora_r, model.lora_alpha, model.lora_dropout,
    model.target_modules
    data.train_data           — *_train_sft_plan.jsonl with question/answer
    data.max_seq_len
    training.lr, .epochs, .batch_size, .grad_accum, .warmup_ratio,
              .weight_decay, .grad_clip, .output_dir

Multi-GPU: launch via `/data/yuyu/.local/bin/torchrun --nproc_per_node=N`
with `HYPPLAN_DIST_BACKEND=gloo` (NCCL is broken on this host per
memory/reference_nccl_topology.md).

Single-GPU: just `python3.10 -m src.train_sft_pt_qwen --config ...`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
)


class QuestionAnswerSFTDataset(Dataset):
    """Reads `{"question": str, "answer": str, ...}` JSONL.

    Builds the chat-template prompt for `question` (with the assistant's
    generation prime appended), then encodes `answer` as the completion.
    Labels mask the prompt tokens (-100) so loss is computed only on
    `answer` tokens (+ EOS)."""

    def __init__(self, tokenizer, data_path: str, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data: list[dict] = []
        with open(data_path) as f:
            for line in f:
                rec = json.loads(line)
                if "question" not in rec or "answer" not in rec:
                    continue
                self.data.append({
                    "question": rec["question"], "answer": rec["answer"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Use the simple "Question: ...\nAnswer: ..." format that
        # `src/eval_pt_ood.py` already uses, so train + eval prompts match.
        prompt_text = f"Question: {item['question']}\nAnswer: "
        completion_text = item["answer"]

        prompt_ids = self.tokenizer.encode(prompt_text,
                                            add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion_text,
                                                add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            completion_ids = list(completion_ids) + [
                self.tokenizer.eos_token_id
            ]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    attn = []
    for b in batch:
        n = len(b["input_ids"])
        pad_n = max_len - n
        input_ids.append(torch.cat([
            b["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)
        ]))
        labels.append(torch.cat([
            b["labels"], torch.full((pad_n,), -100, dtype=torch.long)
        ]))
        attn.append(torch.cat([
            torch.ones(n, dtype=torch.long),
            torch.zeros(pad_n, dtype=torch.long),
        ]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


def setup_distributed():
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        backend = os.environ.get("HYPPLAN_DIST_BACKEND", "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        if backend == "nccl":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend=backend,
                                      timeout=timedelta(hours=8))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, rank, world_size, local_rank, device
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return False, 0, 1, 0, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, rank, world_size, local_rank, device = setup_distributed()
    if rank == 0:
        print(f"world={world_size} device={device}", flush=True)

    torch.manual_seed(args.seed + rank)

    base = config["model"]["base_model"]
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    if rank == 0:
        print(f"Loading {base}", flush=True)
    is_gpt_oss = "gpt-oss" in base.lower()
    if is_gpt_oss:
        # GPT-OSS-20B ships pre-quantized with mxfp4 — let HF auto-handle
        # dtype + placement via device_map="auto". Single-GPU only.
        model = AutoModelForCausalLM.from_pretrained(
            base, trust_remote_code=True, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(device)

    lora_cfg = LoraConfig(
        r=int(config["model"]["lora_r"]),
        lora_alpha=int(config["model"]["lora_alpha"]),
        lora_dropout=float(config["model"]["lora_dropout"]),
        target_modules=list(config["model"]["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    if rank == 0:
        model.print_trainable_parameters()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None,  # gloo wants no device_ids
            output_device=None, find_unused_parameters=False,
        )

    ds = QuestionAnswerSFTDataset(
        tok, config["data"]["train_data"],
        max_seq_len=int(config["data"]["max_seq_len"]),
    )
    if rank == 0:
        print(f"Train records: {len(ds)}", flush=True)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank,
                            shuffle=True, seed=args.seed)
        if distributed else None
    )
    bs = int(config["training"]["batch_size"])
    loader = DataLoader(
        ds, batch_size=bs, sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0, drop_last=True,
    )

    epochs = int(config["training"]["epochs"])
    grad_accum = int(config["training"].get("grad_accum", 1))
    total_steps = (len(loader) // grad_accum) * epochs

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(
            float(config["training"].get("warmup_ratio", 0.05)) * total_steps
        ),
        num_training_steps=total_steps,
    )
    grad_clip = float(config["training"].get("grad_clip", 1.0))

    out_dir = Path(config["training"]["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "config.yaml").open("w") as f:
            yaml.dump(config, f)
    log_path = out_dir / "train.jsonl"

    model.train()
    step = 0
    t0 = time.time()
    if rank == 0:
        print(f"Starting training: total_steps={total_steps}, "
              f"bs={bs}, grad_accum={grad_accum}", flush=True)
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        accum = 0
        optim.zero_grad()
        for batch_i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            accum += 1
            if accum >= grad_accum:
                accum = 0
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
                optim.step()
                scheduler.step()
                optim.zero_grad()
                step += 1
                if rank == 0 and (step % 25 == 0 or step == 1):
                    elapsed = time.time() - t0
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss={out.loss.item():.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed={elapsed:.0f}s", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch, "step": step,
                            "loss": float(out.loss.item()),
                            "elapsed": elapsed,
                        }) + "\n")

    if rank == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.save_pretrained(str(out_dir / "lora"))
        else:
            model.save_pretrained(str(out_dir / "lora"))
        print(f"Saved LoRA to {out_dir / 'lora'}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
