from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.math_grading import extract_boxed, grade_answer
from src.evaluation.math_dataset import load_math_test


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


def build_prompt(problem: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {problem}\n"
        "Solution:\n"
    )


def shard_items(items: list[dict], rank: int, world_size: int) -> list[dict]:
    return [x for i, x in enumerate(items) if i % world_size == rank]


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(int(x["correct"]) for x in results)

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in results:
        level = row.get("level", "unknown")
        subject = row.get("subject", "unknown")
        by_level[level]["total"] += 1
        by_level[level]["correct"] += int(row["correct"])
        by_subject[subject]["total"] += 1
        by_subject[subject]["correct"] += int(row["correct"])

    def finalize(bucket):
        out = {}
        for k, v in bucket.items():
            out[k] = {
                "accuracy": v["correct"] / max(v["total"], 1),
                "correct": v["correct"],
                "total": v["total"],
            }
        return out

    return {
        "overall_accuracy": correct / max(total, 1),
        "overall_correct": correct,
        "overall_total": total,
        "by_level": finalize(by_level),
        "by_subject": finalize(by_subject),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=1.0,
        help="Use only this fraction of the dataset (e.g. 0.1 for 10%%). Default 1.0 = full eval.",
    )
    args = parser.parse_args()

    # Suppress "Setting pad_token_id to eos_token_id" on every generate()
    for _name in ("transformers", "transformers.generation.utils"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    rank, world_size = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    ds = load_math_test()
    all_items = [dict(x) for x in ds]
    if args.subset_ratio < 1.0:
        n = max(1, int(len(all_items) * args.subset_ratio))
        all_items = all_items[:n]
        if rank == 0:
            print(f"Using subset: {n} / {len(ds)} samples ({args.subset_ratio:.0%})")
    shard = shard_items(all_items, rank=rank, world_size=world_size)

    local_results = []
    # Throttle updates so we don't flood the terminal when not a TTY (e.g. under torchrun)
    iterator = tqdm(
        shard,
        disable=(rank != 0),
        mininterval=60.0,
        miniters=50,
    )
    for item in iterator:
        prompt = build_prompt(item["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(out[0], skip_special_tokens=True)

        pred = extract_boxed(generated)
        gt = extract_boxed(item.get("solution", ""))
        correct = grade_answer(pred, gt)

        local_results.append(
            {
                "problem": item["problem"],
                "prediction": pred,
                "ground_truth": gt,
                "correct": correct,
                "level": item.get("level", "unknown"),
                "subject": item.get("type", "unknown"),
            }
        )

    gathered = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
        results = [x for part in gathered for x in part]
    else:
        results = local_results

    if rank == 0:
        metrics = compute_metrics(results)
        payload = {"metrics": metrics, "results": results}
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(metrics, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
