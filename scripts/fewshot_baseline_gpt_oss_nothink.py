"""GPT-OSS-20B fewshot with the analysis (thinking) channel DISABLED.

Approach: after `apply_chat_template(..., add_generation_prompt=True)` which
ends with `<|start|>assistant`, we append the hard channel tag
`<|channel|>final<|message|>`. This forces the model to open directly in the
`final` channel, so it has no opportunity to emit an `analysis` block.

This measures GPT-OSS's raw direct-answer skill, without its reasoning
channel doing the heavy lifting.
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

# Hard-force the assistant turn into the `final` channel so the model can
# never emit the `analysis` (thinking) channel. This is the tightest form of
# "no-thinking" control for GPT-OSS.
FORCE_FINAL_SUFFIX = "<|channel|>final<|message|>"


def build_messages(problem: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for uq, aa in EXAMPLES:
        msgs.append({"role": "user", "content": uq})
        msgs.append({"role": "assistant", "content": aa})
    msgs.append({"role": "user", "content": f"Problem: {problem}"})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--test_data", default="data/24_test_tot.jsonl")
    ap.add_argument("--output",
                    default="results/fewshot_gpt_oss_20b_nothink/generations.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True,
    ).eval()

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
    print(f"fewshot (NO-THINK) on {len(problems)} problems", flush=True)

    t0 = time.time()
    n_correct = 0
    n_format_ok = 0

    with open(args.output, "w") as fout:
        for i, problem in enumerate(problems):
            nums = problem.replace(",", " ")
            msgs = build_messages(nums)
            try:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    reasoning_effort="minimum",
                )
            except TypeError:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            # Force-open the final channel, bypassing any analysis block.
            text = text + FORCE_FINAL_SUFFIX

            enc = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, enc["input_ids"].size(1):],
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
    print(f"\n=== {args.model} fewshot NO-THINK ({n} problems) ===")
    print(f"  accuracy     = {n_correct/n:.4f} ({n_correct}/{n})")
    print(f"  format-valid = {n_format_ok/n:.4f} ({n_format_ok}/{n})")


if __name__ == "__main__":
    main()
