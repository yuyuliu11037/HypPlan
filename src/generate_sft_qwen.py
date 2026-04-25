"""Inference for Qwen-14B SFT on Game-24.

Loads the base model + a LoRA adapter, applies `fewshot_chat_prompt_24`
(same prompt builder used at train time), runs greedy decoding, and writes
generations compatible with [src/evaluate_24.py](src/evaluate_24.py).

Shards the test set across multiple GPUs via `--shard_rank` / `--shard_world`
for data-parallel eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prompt_builders import (
    fewshot_chat_prompt_24, fewshot_chat_prompt_24_plan,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--lora_adapter", required=True,
                    help="Path to a LoRA adapter directory "
                         "(e.g. checkpoints/sft_24_qwen14b/lora).")
    ap.add_argument("--test_data", default="data/24_test_tot.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--prompt_style", choices=["fewshot", "fewshot_plan"],
                    default="fewshot")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading base {args.base_model} in bf16...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device})
    print(f"Attaching LoRA adapter from {args.lora_adapter}...", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_adapter)
    model.eval()

    seen = set()
    records = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item["problem"] not in seen:
                seen.add(item["problem"])
                records.append(item)
    if args.limit > 0:
        records = records[:args.limit]

    records = records[args.shard_rank::args.shard_world]
    print(f"Shard {args.shard_rank}/{args.shard_world}: {len(records)} problems",
          flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            if args.prompt_style == "fewshot_plan":
                prompt_text, add_special = fewshot_chat_prompt_24_plan(tok, rec["problem"])
            else:
                prompt_text, add_special = fewshot_chat_prompt_24(tok, rec["problem"])
            input_ids = tok.encode(
                prompt_text, add_special_tokens=add_special,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out_ids[0, input_ids.size(1):],
                              skip_special_tokens=True)
            # Plan prompt doesn't prime 'Step 1:' — model emits planning token
            # then the step text autoregressively. Fewshot plain prompt does
            # prime 'Step 1:' so we re-prepend it.
            if args.prompt_style == "fewshot_plan":
                gen_full = gen
            else:
                gen_full = "Step 1:" + gen
            fout.write(json.dumps({
                "problem": rec["problem"],
                "ground_truth": rec["text"],
                "generation": gen_full,
            }) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [{args.shard_rank}] {i+1}/{len(records)} ({rate:.2f}/s)",
                      flush=True)

    print(f"Shard {args.shard_rank}: wrote {len(records)} to {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
