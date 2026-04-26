"""Eval Qwen2.5-14B + planning-tokens LoRA on Game-of-24.

Mirrors eval_pt_ood.py for the OOD tasks. Question format matches the PT-SFT
training data (data/24_train_sft_pt.jsonl): "Problem: 8 8 10 13".
Greedy decoding; raw output written for downstream scoring via
src/evaluate_24.parse_and_validate.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_question(rec: dict) -> str:
    pool = rec["problem"].replace(",", " ")
    return f"Problem: {pool}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--lora_adapter", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"Attaching LoRA {args.lora_adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_adapter)
    model.eval()

    seen = set()
    records = []
    for line in open(args.test_data):
        r = json.loads(line)
        p = r["problem"]
        if p in seen:
            continue
        seen.add(p)
        records.append(r)
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval g24 PT-SFT on {len(records)} records "
          f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            q = build_question(rec)
            prompt = f"Question: {q}\nAnswer:"
            input_ids = tok.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out_ids[0, input_ids.size(1):],
                              skip_special_tokens=False)
            fout.write(json.dumps({**rec, "question": q,
                                     "generation": gen}) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                       f"({rate:.2f}/s)", flush=True)
    print(f"  [r{args.shard_rank}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
