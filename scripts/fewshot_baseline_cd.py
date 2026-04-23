"""Few-shot prompting baseline for Countdown on a mid-size Instruct model.

Companion to `scripts/fewshot_baseline.py`, for the Countdown task. Uses the
CD fewshot chat-template prompt (3 exemplars, 6 numbers, variable target)
and evaluates greedy generations with `src.evaluate_cd.parse_and_validate`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prompt_builders import fewshot_chat_prompt_cd
from src.evaluate_cd import parse_and_validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", default="data/cd_test.jsonl")
    ap.add_argument("--output", default="results/fewshot_qwen14b_cd/generations.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load base model in 4-bit NF4 (for 32B on 48GB GPUs).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading {args.model} (4bit={args.load_in_4bit})...", flush=True)
    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, trust_remote_code=True,
            device_map={"": device})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map={"": device})
    model.eval()

    seen = set()
    records = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (tuple(sorted(item["pool"])), item["target"])
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
    if args.limit > 0:
        records = records[:args.limit]
    records = records[args.shard_rank::args.shard_world]
    print(f"Shard {args.shard_rank}/{args.shard_world}: {len(records)} problems",
          flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    n_correct = 0
    n_format_ok = 0
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            prompt_text, add_special = fewshot_chat_prompt_cd(
                tok, rec["pool"], rec["target"])
            input_ids = tok.encode(
                prompt_text, add_special_tokens=add_special,
                return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    pad_token_id=tok.eos_token_id)
            gen = tok.decode(out_ids[0, input_ids.size(1):],
                              skip_special_tokens=True)
            gen_full = "Step 1:" + gen
            n_expected = len(rec["pool"]) - 1
            ok = parse_and_validate(
                rec["pool"], rec["target"], gen_full, n_expected)
            if ok: n_correct += 1
            fout.write(json.dumps({
                "pool": rec["pool"], "target": rec["target"],
                "generation": gen_full, "correct": bool(ok),
            }) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{args.shard_rank}] {i+1}/{len(records)} "
                      f"correct={n_correct} ({rate:.2f}/s)",
                      flush=True)

    summary = {
        "model": args.model, "n": len(records),
        "accuracy": n_correct / max(1, len(records)),
        "n_correct": n_correct,
    }
    sumpath = os.path.join(os.path.dirname(args.output), f"metrics_shard{args.shard_rank}.json")
    with open(sumpath, "w") as fs:
        json.dump(summary, fs, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
