"""Few-shot prompting baseline for Game-of-24 on a mid-size Instruct model.

Decides whether a given model can serve as a base for HypPlan without SFT:
- If greedy accuracy is high enough (say ≥10%), DAgger stage 2 has bootstrap
  signal and we can skip SFT entirely.
- If it's 0-5%, SFT is still required for bootstrap.

Uses the model's chat template if available, with 3 full-trajectory exemplars
in our "Step N: a op b = r. Remaining: ..." format so evaluate_24 can parse
the model's output directly.
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
    (
        "Problem: 4 4 6 8",
        "Step 1: 4 + 8 = 12. Remaining: 4 6 12\n"
        "Step 2: 6 - 4 = 2. Remaining: 2 12\n"
        "Step 3: 2 * 12 = 24. Answer: 24",
    ),
    (
        "Problem: 2 9 10 12",
        "Step 1: 12 * 2 = 24. Remaining: 9 10 24\n"
        "Step 2: 10 - 9 = 1. Remaining: 1 24\n"
        "Step 3: 24 * 1 = 24. Answer: 24",
    ),
    (
        "Problem: 4 9 10 13",
        "Step 1: 13 - 10 = 3. Remaining: 3 4 9\n"
        "Step 2: 9 - 3 = 6. Remaining: 4 6\n"
        "Step 3: 4 * 6 = 24. Answer: 24",
    ),
]


def build_chat(tokenizer, problem: str) -> str:
    """Build a chat-templated prompt with few-shot exemplars."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for user_q, assistant_a in EXAMPLES:
        msgs.append({"role": "user", "content": user_q})
        msgs.append({"role": "assistant", "content": assistant_a})
    msgs.append({"role": "user", "content": f"Problem: {problem}"})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", default="data/24_test_tot.jsonl")
    ap.add_argument("--output", default="results/fewshot_qwen14b/generations.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
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
    # Data-parallel sharding: each rank handles its slice of the problem list
    if args.shard_world > 1:
        problems = problems[args.shard_rank :: args.shard_world]
    print(f"Rank {args.shard_rank}/{args.shard_world}: "
          f"running few-shot on {len(problems)} problems", flush=True)

    t0 = time.time()
    n_valid = 0
    n_correct = 0
    n_format_ok = 0

    with open(args.output, "w") as fout:
        for start in range(0, len(problems), args.batch_size):
            chunk = problems[start: start + args.batch_size]
            nums_strs = [p.replace(",", " ") for p in chunk]
            prompts = [build_chat(tok, nums) for nums in nums_strs]
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

            for problem, gen in zip(chunk, decoded):
                # Prepend "Step 1:" parity doesn't apply — model emits full
                # trajectory. Validate directly.
                valid = parse_and_validate(problem, gen)
                # Format OK = contains three "Step N:" lines
                fmt_ok = gen.count("Step 1:") >= 1 and gen.count("Step 3:") >= 1
                n_correct += int(valid)
                n_valid += int(valid)
                n_format_ok += int(fmt_ok)
                fout.write(json.dumps({
                    "problem": problem, "generation": gen,
                    "valid": valid, "format_ok": fmt_ok,
                }) + "\n")
                fout.flush()

            elapsed = time.time() - t0
            done = min(start + args.batch_size, len(problems))
            print(f"  {done}/{len(problems)}  acc={n_correct/max(done,1):.3f}"
                  f"  format_ok={n_format_ok/max(done,1):.3f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    n = len(problems)
    print(f"\n=== {args.model} few-shot on {n} problems ===")
    print(f"  accuracy      = {n_correct/n:.4f} ({n_correct}/{n})")
    print(f"  format-valid  = {n_format_ok/n:.4f} ({n_format_ok}/{n})")


if __name__ == "__main__":
    main()
