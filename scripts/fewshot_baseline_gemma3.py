"""Gemma-3-27B-it few-shot baseline for Game-of-24.

Needs bf16 (4-bit NF4 produces NaN logits for Gemma family). 27B bf16 is
~54GB, split across 2 GPUs via device_map="auto".

Uses AutoProcessor + multimodal-style content
(`[{"type":"text", "text":"..."}]`) because Gemma3 is loaded as
`Gemma3ForConditionalGeneration` and its processor expects that shape.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate_24 import parse_and_validate


SYSTEM = (
    "You are a careful arithmetic solver. Use the four given numbers and "
    "basic arithmetic operations (+, -, *, /) to obtain 24. Each number "
    "must be used exactly once. Respond with exactly three lines in the "
    "format shown in the examples, ending with 'Answer: 24' on the last "
    "step. Do not add any other text."
)

EXAMPLES = [
    ("Problem: 4 4 6 8",
     "Step 1: 4 + 8 = 12. Remaining: 4 6 12\n"
     "Step 2: 6 - 4 = 2. Remaining: 2 12\n"
     "Step 3: 2 * 12 = 24. Answer: 24"),
    ("Problem: 2 9 10 12",
     "Step 1: 12 * 2 = 24. Remaining: 9 10 24\n"
     "Step 2: 10 - 9 = 1. Remaining: 1 24\n"
     "Step 3: 24 * 1 = 24. Answer: 24"),
    ("Problem: 4 9 10 13",
     "Step 1: 13 - 10 = 3. Remaining: 3 4 9\n"
     "Step 2: 9 - 3 = 6. Remaining: 4 6\n"
     "Step 3: 4 * 6 = 24. Answer: 24"),
]


def build_messages(problem: str) -> list[dict]:
    msgs = [{"role": "system",
             "content": [{"type": "text", "text": SYSTEM}]}]
    for uq, aa in EXAMPLES:
        msgs.append({"role": "user",
                     "content": [{"type": "text", "text": uq}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": aa}]})
    msgs.append({"role": "user",
                 "content": [{"type": "text", "text": f"Problem: {problem}"}]})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--test_data", default="data/24_test_tot.jsonl")
    ap.add_argument("--output",
                    default="results/fewshot_gemma3_27b/generations.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading {args.model} bf16 across visible GPUs", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    eos_id = proc.tokenizer.eos_token_id if hasattr(proc, "tokenizer") else model.config.eos_token_id

    seen: set = set()
    problems: list[str] = []
    with open(args.test_data) as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                problems.append(p)
    if args.limit > 0:
        problems = problems[: args.limit]
    print(f"fewshot on {len(problems)} problems", flush=True)

    t0 = time.time()
    n_correct = 0
    n_format_ok = 0

    with open(args.output, "w") as fout:
        for i, problem in enumerate(problems):
            nums = problem.replace(",", " ")
            msgs = build_messages(nums)
            inputs = proc.apply_chat_template(
                msgs, tokenize=True, return_tensors="pt",
                add_generation_prompt=True, return_dict=True,
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    pad_token_id=eos_id,
                )
            gen = proc.decode(out[0, inputs["input_ids"].size(1):],
                               skip_special_tokens=True)

            valid = parse_and_validate(problem, gen)
            fmt_ok = gen.count("Step 1:") >= 1 and gen.count("Step 3:") >= 1
            n_correct += int(valid)
            n_format_ok += int(fmt_ok)
            fout.write(json.dumps({
                "problem": problem, "generation": gen,
                "valid": valid, "format_ok": fmt_ok,
            }) + "\n")
            fout.flush()

            done = i + 1
            if done % 5 == 0 or done == len(problems):
                elapsed = time.time() - t0
                print(f"  {done}/{len(problems)}  acc={n_correct/done:.3f}"
                      f"  fmt={n_format_ok/done:.3f}  elapsed={elapsed:.0f}s",
                      flush=True)

    n = len(problems)
    print(f"\n=== {args.model} fewshot ({n} problems) ===")
    print(f"  accuracy     = {n_correct/n:.4f} ({n_correct}/{n})")
    print(f"  format-valid = {n_format_ok/n:.4f} ({n_format_ok}/{n})")


if __name__ == "__main__":
    main()
