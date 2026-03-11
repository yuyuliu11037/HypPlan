from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.dataset import ReasoningPath, get_rank_shard_indices, load_gsm8k_aug_dataset
from src.model.planning_head import LoraSettings, load_tokenizer_and_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA-SFT baseline without planning vectors.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/sft-baseline")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_prompt(question: str) -> str:
    return (
        "You are a math reasoning assistant.\n"
        "Solve the question step by step and end with the final answer.\n\n"
        f"Question:\n{question}\n\n"
        "Reasoning:\n"
    )


def build_target(sample: ReasoningPath) -> str:
    steps_text = "\n".join(sample.steps)
    return f"{steps_text}\nFinal answer: {sample.answer}"


def load_data(cfg: Dict[str, Any]) -> List[ReasoningPath]:
    data = load_gsm8k_aug_dataset(
        dataset_name=cfg["dataset_name"],
        split=cfg.get("train_split", "train"),
        max_samples=cfg.get("max_train_samples"),
    )
    ratio = float(cfg.get("train_subset_ratio", 1.0))
    seed = int(cfg.get("train_subset_seed", cfg.get("seed", 42)))
    if 0.0 < ratio < 1.0:
        keep = max(1, int(len(data) * ratio))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(data), generator=g)[:keep].tolist()
        data = [data[i] for i in idx]
    return data


def build_batch_tensors(
    tokenizer,
    batch: List[ReasoningPath],
    max_prompt_len: int,
    max_target_len: int,
    device: torch.device,
):
    prompts = [build_prompt(s.question) for s in batch]
    targets = [build_target(s) for s in batch]
    prompt_tok = tokenizer(
        prompts,
        truncation=True,
        max_length=max_prompt_len,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    target_tok = tokenizer(
        targets,
        truncation=True,
        max_length=max_target_len,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_tok.input_ids.to(device)
    prompt_mask = prompt_tok.attention_mask.to(device)
    target_ids = target_tok.input_ids.to(device)
    target_mask = target_tok.attention_mask.to(device)

    empty_targets = target_mask.sum(dim=1) == 0
    if torch.any(empty_targets):
        target_ids[empty_targets, 0] = tokenizer.eos_token_id
        target_mask[empty_targets, 0] = 1

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, target_mask], dim=1)
    labels = torch.full_like(input_ids, -100)
    prompt_pad = prompt_ids.size(1)
    for b in range(input_ids.size(0)):
        t_len = int(target_mask[b].sum().item())
        labels[b, prompt_pad : prompt_pad + t_len] = target_ids[b, :t_len]
    return input_ids, attention_mask, labels


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg["mixed_precision"],
    )
    torch.manual_seed(int(cfg["seed"]))

    lora_settings = LoraSettings(
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    tokenizer, model = load_tokenizer_and_model(cfg["model_name_or_path"], lora_settings, device_map=None)
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    model, optimizer = accelerator.prepare(model, optimizer)
    data = load_data(cfg)
    shard_indices = get_rank_shard_indices(
        total_size=len(data),
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    local_data = [data[i] for i in shard_indices]

    bs = max(1, int(cfg["per_device_batch_size"]))
    local_num_batches = math.ceil(len(local_data) / bs) if local_data else 0
    local_num_batches_tensor = torch.tensor([local_num_batches], device=accelerator.device, dtype=torch.int64)
    gathered_num_batches = accelerator.gather(local_num_batches_tensor)
    max_num_batches = int(gathered_num_batches.max().item())
    if args.max_steps is not None:
        max_num_batches = min(max_num_batches, args.max_steps)

    dummy_sample: ReasoningPath | None = None
    if max_num_batches > 0:
        if local_data:
            dummy_sample = local_data[0]
        elif data:
            # Keep collectives aligned across ranks with a deterministic dummy batch.
            dummy_sample = data[0]
        else:
            raise RuntimeError("No training samples available for synchronized distributed steps.")

    model.train()
    for epoch in range(args.epochs):
        iterator = tqdm(
            range(max_num_batches),
            desc=f"SFT epoch={epoch} rank={accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for bidx in iterator:
            start = bidx * bs
            is_real_batch = start < len(local_data)
            if is_real_batch:
                batch = local_data[start : start + bs]
            else:
                if dummy_sample is None:
                    raise RuntimeError("Missing dummy batch sample for synchronized distributed step.")
                batch = [dummy_sample]
            with accelerator.accumulate(model):
                input_ids, attention_mask, labels = build_batch_tensors(
                    tokenizer=tokenizer,
                    batch=batch,
                    max_prompt_len=cfg["max_question_tokens"],
                    max_target_len=cfg["max_step_tokens"] * 8,
                    device=accelerator.device,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss
                if not is_real_batch:
                    # Keep backward/collective ordering identical across ranks.
                    loss = loss * 0.0
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(out_dir / "lora_adapter")
        tokenizer.save_pretrained(out_dir / "tokenizer")
        accelerator.print(f"Saved SFT baseline adapter to {out_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()
