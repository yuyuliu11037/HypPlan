"""Baseline inference: plain Qwen2.5-7B generation (frozen, no planning)."""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.multiprocessing as mp
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(config: dict, device: str = "cuda"):
    """Load the frozen base model."""
    model_name = config["model"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, problem: str, max_new_tokens: int = 2048,
             temperature: float = 0.0, device: str = "cuda") -> str:
    """Standard autoregressive generation with the base model."""
    input_ids = tokenizer.encode(problem, return_tensors="pt").to(device)

    if temperature <= 0:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_ids.size(1):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _worker(rank, world_size, config, args, records):
    """Worker for multi-GPU inference."""
    device = f"cuda:{rank}"
    model, tokenizer = load_model(config, device)

    shard = [r for i, r in enumerate(records) if i % world_size == rank]

    results = []
    for i, record in enumerate(shard):
        generation = generate(
            model, tokenizer, record["problem"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        results.append({
            "idx": record["_idx"],
            "problem": record["problem"],
            "solution": record["solution"],
            "generation": generation,
            "level": record.get("level", ""),
            "type": record.get("type", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"[GPU {rank}] Generated {i+1}/{len(shard)}")

    shard_path = args.output + f".shard{rank}"
    with open(shard_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[GPU {rank}] Done — {len(results)} generations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max test samples (0=use config eval_samples)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load test data
    from datasets import load_dataset, concatenate_datasets
    all_ds = []
    for cfg_name in config["data"]["configs"]:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg_name, split="test")
        all_ds.append(ds)
    test_data = concatenate_datasets(all_ds)

    max_samples = args.max_samples or config["data"].get("eval_samples", len(test_data))
    if max_samples < len(test_data):
        test_data = test_data.shuffle(seed=42).select(range(max_samples))

    records = []
    for i in range(len(test_data)):
        item = test_data[i]
        item["_idx"] = i
        records.append(item)

    num_gpus = args.num_gpus or torch.cuda.device_count()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if num_gpus <= 1:
        _worker(0, 1, config, args, records)
    else:
        print(f"Launching {num_gpus}-GPU parallel inference")
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=_worker, args=(rank, num_gpus, config, args, records))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # Merge shards
    all_results = []
    for rank in range(num_gpus):
        shard_path = args.output + f".shard{rank}"
        if os.path.exists(shard_path):
            with open(shard_path) as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(shard_path)

    all_results.sort(key=lambda r: r["idx"])
    with open(args.output, "w") as f:
        for r in all_results:
            del r["idx"]
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(all_results)} generations to {args.output}")


if __name__ == "__main__":
    main()
