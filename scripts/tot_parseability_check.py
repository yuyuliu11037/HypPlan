"""Quick parseability check for ToT-style propose prompts on a mid-size model.

For each of the first N test problems, issue the ToT propose_prompt at the
root state (initial 4 numbers). Count:
  - how many candidates parse from the model's output
  - of those, how many are pool-consistent (prev − {a,b} ∪ {r} == new_remaining)

If parse_rate is low, the model can't follow ToT's propose format.
If pool_correct_rate is low, the model hallucinates remaining numbers (the
failure mode we saw with our 8B-SFT model).

Prints a per-problem breakdown + aggregates. No GPU training, no BFS — just
one propose call per problem.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tot_baseline import PROPOSE_PROMPT, parse_candidates


def pool_consistent(prev_nums: list[str], a: str, b: str, r: str,
                    new_nums: list[str]) -> bool:
    """Check new_nums == prev_nums - {a, b} + {r} as multisets.

    All values compared as strings to avoid float parsing issues.
    """
    prev = list(prev_nums)
    try:
        prev.remove(a)
    except ValueError:
        return False
    try:
        prev.remove(b)
    except ValueError:
        return False
    prev.append(r)
    return sorted(prev) == sorted(new_nums)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_data", default="data/24_test_tot.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    # Load first N unique problems
    seen: set = set()
    problems: list[str] = []
    with open(args.test_data) as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                problems.append(p)
            if len(problems) >= args.n:
                break

    total_cands = 0
    total_pool_correct = 0
    n_problems_with_any_candidate = 0
    n_problems_with_any_pool_correct = 0

    for i, problem in enumerate(problems):
        initial = problem.replace(",", " ")
        prev_nums = initial.split()
        prompt = PROPOSE_PROMPT.format(input=initial)
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else 1.0,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0, enc["input_ids"].size(1):],
                         skip_special_tokens=True)
        # DeepSeek R1-distill emits <think>...</think>; try to parse the bit
        # after the thought block if present, else the whole thing.
        post_think = re.split(r"</think>\s*", gen, maxsplit=1)
        tail = post_think[-1]
        cands = parse_candidates(tail)
        # If nothing in tail, also try the full output
        if not cands:
            cands = parse_candidates(gen)

        pool_correct = 0
        for c in cands:
            # Re-extract a/op/b/r from the reconstructed line
            m = re.match(
                r"(-?[\d./]+)\s*([+\-*/])\s*(-?[\d./]+)\s*=\s*(-?[\d./]+)"
                r"\s*\(left:\s*(.*)\)", c["line"])
            if not m:
                continue
            a, _op, b, r, left = m.groups()
            new_nums = left.strip().split()
            if pool_consistent(prev_nums, a, b, r, new_nums):
                pool_correct += 1

        total_cands += len(cands)
        total_pool_correct += pool_correct
        if cands:
            n_problems_with_any_candidate += 1
        if pool_correct > 0:
            n_problems_with_any_pool_correct += 1

        print(f"[{i+1}/{len(problems)}] problem={problem}  "
              f"parsed={len(cands)}  pool_correct={pool_correct}",
              flush=True)
        if len(cands) == 0:
            print(f"  raw output (first 300 chars): {gen[:300]!r}")
        elif pool_correct == 0:
            print(f"  first candidate: {cands[0]['line']}")
            print(f"    expected pool ops: prev={prev_nums}")

    n = len(problems)
    print(f"\n=== Summary (N={n}) ===")
    print(f"  total candidates parsed: {total_cands} ({total_cands/n:.1f}/problem)")
    print(f"  pool-correct candidates: {total_pool_correct} "
          f"({total_pool_correct/max(total_cands,1)*100:.1f}% of parsed)")
    print(f"  problems with ≥1 parsed candidate:    "
          f"{n_problems_with_any_candidate}/{n}")
    print(f"  problems with ≥1 pool-correct parse:  "
          f"{n_problems_with_any_pool_correct}/{n}")


if __name__ == "__main__":
    main()
