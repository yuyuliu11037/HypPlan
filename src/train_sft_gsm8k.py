"""SFT trainer for Phi-1.5 on GSM8K (baseline or planning-token variant).

Mirrors the setup used in the Planning Tokens paper (Wang et al. 2023):
- Phi-1.5 base, LoRA r=16, lr=2e-4, AdamW, 10 epochs.
- Completion-only loss: prompt is "Question: ...\\nAnswer: ", labels are
  -100 over the prompt and the actual token ids over the completion.
- Bf16 if available.

Single GPU, no DDP needed (Phi-1.5 is 1.3B and fits in <12 GB bf16).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from datetime import timedelta
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


def _setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        backend = os.environ.get("HYPPLAN_DIST_BACKEND", "gloo")
        dist.init_process_group(backend=backend, timeout=timedelta(hours=8))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return distributed, device, rank, world_size, local_rank


def _all_reduce_grads(params, world_size):
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


PROMPT_TEMPLATE = "Question: {q}\nAnswer:"


class GSM8KSFTDataset(Dataset):
    def __init__(self, tokenizer, data_path: str, max_seq_len: int = 512):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.data = [json.loads(l) for l in open(data_path)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        prompt = PROMPT_TEMPLATE.format(q=rec["question"])
        completion = " " + rec["answer"]
        prompt_ids = self.tok.encode(prompt, add_special_tokens=True)
        completion_ids = self.tok.encode(completion, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            completion_ids.append(self.tok.eos_token_id)
        input_ids = prompt_ids + completion_ids
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        labels = [-100] * len(prompt_ids) + completion_ids
        labels = labels[: len(input_ids)]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, device, rank, world_size, _ = _setup_distributed()
    torch.manual_seed(args.seed + rank)

    model_name = config["model"]["base_model"]
    if rank == 0:
        print(f"Loading {model_name} (world_size={world_size})", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        target_modules=config["model"].get("target_modules",
                                              ["q_proj", "k_proj",
                                               "v_proj", "dense"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if rank == 0:
        model.print_trainable_parameters()

    ds = GSM8KSFTDataset(tok, config["data"]["train_data"],
                          max_seq_len=config["data"].get("max_seq_len", 512))
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size,
                                       rank=rank, shuffle=True,
                                       seed=args.seed)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"],
                         shuffle=(sampler is None), sampler=sampler,
                         collate_fn=lambda b: collate(b, tok.pad_token_id),
                         num_workers=2, pin_memory=True, drop_last=True)

    epochs = int(config["training"]["epochs"])
    grad_accum = int(config["training"].get("grad_accum", 1))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    total_steps = (len(loader) * epochs) // grad_accum
    warmup = int(total_steps * float(config["training"].get("warmup_ratio", 0.05)))
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                       lr=float(config["training"]["lr"]),
                       weight_decay=float(config["training"].get("weight_decay", 0.0)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir = Path(config["training"]["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "config.yaml").open("w") as f:
            yaml.dump(config, f)
    log_path = out_dir / "train.jsonl"
    trainable = [p for p in model.parameters() if p.requires_grad]

    step = 0
    t0 = time.time()
    accum_loss, n_loss = 0.0, 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += float(loss.item()); n_loss += 1
            if (bi + 1) % grad_accum == 0:
                if distributed:
                    _all_reduce_grads(trainable, world_size)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if rank == 0 and step % 50 == 0:
                    avg = accum_loss / max(n_loss, 1) * grad_accum
                    print(f"step {step}/{total_steps} | epoch {epoch} | "
                           f"loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}",
                           flush=True)
                    with log_path.open("a") as f:
                        f.write(json.dumps({"step": step, "epoch": epoch,
                                              "loss": round(avg, 4)}) + "\n")
                    accum_loss, n_loss = 0.0, 0
        if rank == 0:
            print(f"== epoch {epoch} done in {time.time()-t0:.0f}s ==", flush=True)
        t0 = time.time()
    if rank == 0:
        model.save_pretrained(out_dir / "lora")
        print(f"Saved LoRA to {out_dir/'lora'}", flush=True)


if __name__ == "__main__":
    main()
