from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.math_grading import extract_final_answer, grade_answer


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n"
    "Solution:\n"
)


def ddp_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def shard(items: list[dict], rank: int, world_size: int) -> list[dict]:
    return [x for i, x in enumerate(items) if i % world_size == rank]


def summarize(results: list[dict]) -> dict:
    overall = sum(int(r["correct"]) for r in results) / max(1, len(results))
    by_level = defaultdict(list)
    by_subject = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(int(r["correct"]))
        by_subject[r["subject"]].append(int(r["correct"]))
    return {
        "overall_accuracy": overall,
        "count": len(results),
        "by_level": {k: sum(v) / len(v) for k, v in by_level.items()},
        "by_subject": {k: sum(v) / len(v) for k, v in by_subject.items()},
    }


def load_local_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_math_test_split(dataset_name: str, split: str, dataset_config: str | None):
    tried: list[tuple[str, str | None, str]] = []
    candidates = [dataset_name]
    for fallback in ["EleutherAI/hendrycks_math", "hendrycks/competition_math"]:
        if fallback not in candidates:
            candidates.append(fallback)

    for name in candidates:
        config = dataset_config if name == dataset_name else None
        try:
            if config:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            print(f"Loaded dataset: name={name}, split={split}, config={config}")
            return ds
        except Exception as exc:  # noqa: BLE001
            tried.append((name, config, str(exc)))

    details = "\n".join(
        f"- name={name}, split={split}, config={config}: {err}" for name, config, err in tried
    )
    raise RuntimeError(
        "Failed to load MATH dataset from provided name and fallbacks.\n"
        f"Tried:\n{details}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset_name", default="EleutherAI/hendrycks_math")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--local_eval_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--lora_adapter_path", default=None)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if args.lora_adapter_path:
        model = PeftModel.from_pretrained(base, args.lora_adapter_path)
    else:
        model = base
    model = model.to(device)
    model.eval()

    if args.local_eval_path:
        problems = load_local_jsonl(args.local_eval_path)
        if rank == 0:
            print(f"Loaded local eval file: {args.local_eval_path} ({len(problems)} samples)")
    else:
        ds = load_math_test_split(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            dataset_config=args.dataset_config,
        )
        problems = [dict(x) for x in ds]
    if args.max_samples is not None:
        problems = problems[: args.max_samples]
    local = shard(problems, rank, world_size)

    local_results = []
    for item in local:
        prompt = PROMPT_TEMPLATE.format(problem=item["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = out[0][inputs["input_ids"].size(1) :]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        pred = extract_final_answer(completion_text)
        if "solution" in item:
            gold = extract_final_answer(str(item["solution"])) or str(item["solution"]).strip()
        else:
            gold = str(item.get("ground_truth_answer", "")).strip()
        local_results.append(
            {
                "problem": item["problem"],
                "subject": item.get("type", item.get("subject", "unknown")),
                "level": str(item.get("level", "unknown")),
                "prediction": pred,
                "gold": gold,
                "correct": grade_answer(pred, gold),
                "generated_tokens": int(completion_ids.numel()),
                "generated_text": completion_text,
            }
        )

    gathered: list[list[dict]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]

    if rank == 0:
        merged = [x for part in gathered for x in part]
        payload = {"metrics": summarize(merged), "results": merged}
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(payload["metrics"], indent=2))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

