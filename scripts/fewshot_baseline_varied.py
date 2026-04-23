"""Few-shot baseline for varied-target Game-of-24.

Reads {pool, target, n_steps} records from the varied-target test jsonl
produced by [data/generate_24_varied.py](data/generate_24_varied.py), builds
prompts with the generic chat template, runs greedy decoding, and validates
via [src/evaluate_generic.py](src/evaluate_generic.py).

Supports bf16 and 4-bit (--load_in_4bit) loading so we can test both
Qwen2.5-14B and Qwen2.5-32B on the same GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate_generic import parse_and_validate_generic
from src.prompt_builders import fewshot_chat_prompt_generic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_data", default="data/24_varied_test.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model} (4bit={args.load_in_4bit})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()

    records: list[dict] = []
    with open(args.test_data) as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    # Data-parallel shard: each rank takes its slice
    if args.shard_world > 1:
        records = records[args.shard_rank:: args.shard_world]
    print(f"Rank {args.shard_rank}/{args.shard_world}: running varied-target "
          f"fewshot on {len(records)} records", flush=True)

    t0 = time.time()
    per_depth_correct: dict[int, int] = defaultdict(int)
    per_depth_total: dict[int, int] = defaultdict(int)
    n_correct = 0

    with open(args.output, "w") as fout:
        for start in range(0, len(records), args.batch_size):
            chunk = records[start: start + args.batch_size]
            prompts = [fewshot_chat_prompt_generic(tok, r["pool"], r["target"])[0]
                       for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    pad_token_id=tok.pad_token_id,
                )
            gen_tokens = out[:, enc["input_ids"].size(1):]
            decoded = tok.batch_decode(gen_tokens, skip_special_tokens=True)

            for rec, gen in zip(chunk, decoded):
                valid = parse_and_validate_generic(rec["pool"], rec["target"],
                                                   gen)
                n_correct += int(valid)
                per_depth_total[rec["n_steps"]] += 1
                per_depth_correct[rec["n_steps"]] += int(valid)
                fout.write(json.dumps({
                    "pool": rec["pool"], "target": rec["target"],
                    "n_steps": rec["n_steps"],
                    "generation": gen, "valid": valid,
                }) + "\n")
                fout.flush()

            done = min(start + args.batch_size, len(records))
            if done % 40 == 0 or done == len(records):
                elapsed = time.time() - t0
                print(f"  {done}/{len(records)}  acc={n_correct/max(done,1):.3f}"
                      f"  elapsed={elapsed:.0f}s", flush=True)

    n = len(records)
    print(f"\n=== {args.model} varied-target fewshot ({n} records) ===")
    print(f"  overall accuracy = {n_correct/n:.4f} ({n_correct}/{n})")
    for d in sorted(per_depth_total.keys()):
        tot, cor = per_depth_total[d], per_depth_correct[d]
        print(f"  depth={d}: {cor/max(tot,1):.4f} ({cor}/{tot})")


if __name__ == "__main__":
    main()
