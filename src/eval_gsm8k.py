"""Eval Phi-1.5 + LoRA on GSM8K test set.

Greedy decoding from "Question: ...\\nAnswer: ", then extract the final
integer (`#### N`). Compares to the gold integer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "Question: {q}\nAnswer:"
FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def extract_final(gen: str):
    m = FINAL_RE.search(gen)
    if not m:
        return None
    s = m.group(1)
    return float(s) if "." in s else int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="microsoft/phi-1_5")
    ap.add_argument("--lora_adapter", default=None,
                     help="If omitted, eval the base model.")
    ap.add_argument("--test_data", default="data/gsm8k_test.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"Loading {args.base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    if args.lora_adapter:
        print(f"Attaching LoRA {args.lora_adapter}", flush=True)
        model = PeftModel.from_pretrained(base, args.lora_adapter)
    else:
        model = base
    model.eval()

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval on {len(records)} records "
           f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_correct = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            prompt = PROMPT_TEMPLATE.format(q=rec["question"])
            input_ids = tok.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out_ids[0, input_ids.size(1):],
                              skip_special_tokens=True)
            pred = extract_final(gen)
            ok = (pred is not None and rec["final"] is not None
                   and float(pred) == float(rec["final"]))
            n_correct += int(ok)
            fout.write(json.dumps({
                "question": rec["question"], "gold": rec["final"],
                "pred": pred, "ok": ok, "generation": gen,
            }) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                       f"acc={n_correct/(i+1):.4f} ({rate:.2f}/s)",
                       flush=True)
    print(f"FINAL: {n_correct}/{len(records)} = "
           f"{n_correct/len(records):.4f}", flush=True)


if __name__ == "__main__":
    main()
