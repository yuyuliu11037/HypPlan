"""Generate DPO preference pairs for planning vector training on Game of 24.

For each (problem, successful trajectory, context k), emit a (ctx, r+, r-) triple:
- r+ = next step from the successful trajectory
- r- = a different next-step operation whose resulting state cannot reach 24

Contexts k=0 and k=1 only (exclude k=2 to avoid OOD "Answer: X" negatives).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from fractions import Fraction
from itertools import combinations

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DATA_DIR, ".."))
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, REPO_ROOT)

from generate_24_trajectories import OPS, solve
from src.dataset_24 import make_prompt


_SOLV_CACHE: dict = {}


def is_solvable(nums) -> bool:
    key = tuple(sorted(nums))
    if key not in _SOLV_CACHE:
        _SOLV_CACHE[key] = bool(solve(key))
    return _SOLV_CACHE[key]


def fraction_to_str(f: Fraction) -> str:
    return str(int(f)) if f.denominator == 1 else str(f)


def format_step_body(a, op, b, r, step_num, new_pool, is_last=False):
    """'Step N: a op b = r. Remaining: ...' or 'Step N: ... Answer: 24'."""
    a_s, b_s, r_s = fraction_to_str(a), fraction_to_str(b), fraction_to_str(r)
    body = f"Step {step_num}: {a_s} {op} {b_s} = {r_s}."
    if is_last:
        body += " Answer: 24"
    else:
        remaining = " ".join(fraction_to_str(x) for x in sorted(new_pool))
        body += f" Remaining: {remaining}"
    return body


def format_step_tail(a, op, b, r, new_pool, is_last=False):
    """Content after 'Step N:' marker — r+ or r- string."""
    a_s, b_s, r_s = fraction_to_str(a), fraction_to_str(b), fraction_to_str(r)
    tail = f" {a_s} {op} {b_s} = {r_s}."
    if is_last:
        tail += " Answer: 24"
    else:
        remaining = " ".join(fraction_to_str(x) for x in sorted(new_pool))
        tail += f" Remaining: {remaining}"
    return tail


def canon_op(a, op, b):
    if op in ('+', '*'):
        return (op, tuple(sorted([a, b])))
    return (op, (a, b))


STEP_RE = re.compile(
    r'Step\s+\d+:\s*(-?\d+(?:/\d+)?)\s+([+\-*/])\s+(-?\d+(?:/\d+)?)\s*=\s*(-?\d+(?:/\d+)?)'
)


def parse_trajectory(text, problem_numbers):
    """Parse trajectory text into ordered step dicts with pool bookkeeping."""
    matches = list(STEP_RE.finditer(text))
    if len(matches) != 3:
        return None
    steps = []
    pool = [Fraction(n) for n in problem_numbers]
    for i, m in enumerate(matches):
        a = Fraction(m.group(1))
        op = m.group(2)
        b = Fraction(m.group(3))
        r = Fraction(m.group(4))
        new_pool = list(pool)
        try:
            new_pool.remove(a)
            new_pool.remove(b)
        except ValueError:
            return None
        new_pool.append(r)
        steps.append({
            "a": a, "op": op, "b": b, "r": r,
            "pool_before": list(pool),
            "pool_after": list(new_pool),
            "step_num": i + 1,
            "is_last": (i == len(matches) - 1),
        })
        pool = new_pool
    return steps


def enumerate_ops(pool):
    """Yield (a, op, b, result, new_pool) for all valid binary ops over pool."""
    for i, j in combinations(range(len(pool)), 2):
        a, b = pool[i], pool[j]
        remaining = [pool[k] for k in range(len(pool)) if k != i and k != j]
        for op_sym, op_fn, commutative in OPS:
            orderings = [(a, b)] if commutative else [(a, b), (b, a)]
            for x, y in orderings:
                result = op_fn(x, y)
                if result is None:
                    continue
                new_pool = remaining + [result]
                yield x, op_sym, y, result, new_pool


def main():
    INPUT = os.path.join(DATA_DIR, "24_train_plan5k_tot.jsonl")
    OUTPUT = os.path.join(DATA_DIR, "24_train_dpo_tot.jsonl")
    N_TRAJS_PER_PROB = 2
    SEED = 42
    LEN_DIFF_CAP = 3

    random.seed(SEED)

    by_problem = defaultdict(list)
    with open(INPUT) as f:
        for line in f:
            item = json.loads(line)
            by_problem[item["problem"]].append(item)
    total_trajs = sum(len(v) for v in by_problem.values())
    print(f"Loaded {total_trajs} trajectories across {len(by_problem)} problems")

    pairs = []
    skipped_no_neg = 0
    sanity_fail = 0

    for problem, trajs in by_problem.items():
        nums = [int(x) for x in problem.split(",")]
        chosen = random.sample(trajs, min(N_TRAJS_PER_PROB, len(trajs)))

        for traj_entry in chosen:
            steps = parse_trajectory(traj_entry["text"], nums)
            if steps is None:
                sanity_fail += 1
                continue

            for k in (0, 1):
                state = steps[k]["pool_before"]
                pos_canon = canon_op(steps[k]["a"], steps[k]["op"], steps[k]["b"])

                # Enumerate alternative operations and classify
                negs = []
                for x, op, y, result, new_pool in enumerate_ops(state):
                    if canon_op(x, op, y) == pos_canon:
                        continue
                    if not is_solvable(new_pool):
                        negs.append((x, op, y, result, new_pool))

                if not negs:
                    skipped_no_neg += 1
                    continue

                neg = random.choice(negs)

                # r+ and r- tails (content after "Step N:")
                pos_tail = format_step_tail(
                    steps[k]["a"], steps[k]["op"], steps[k]["b"], steps[k]["r"],
                    steps[k]["pool_after"], is_last=steps[k]["is_last"],
                )
                neg_tail = format_step_tail(
                    neg[0], neg[1], neg[2], neg[3], neg[4], is_last=False,
                )

                # Build context text
                prompt = make_prompt(problem)  # ends with "Step 1:"
                if k == 0:
                    ctx_text = prompt
                else:  # k == 1
                    s1 = steps[0]
                    s1_body = format_step_body(
                        s1["a"], s1["op"], s1["b"], s1["r"],
                        step_num=1, new_pool=s1["pool_after"], is_last=False,
                    )
                    # Need: prompt (ends with "Step 1:") + tail of s1 + "\nStep 2:"
                    # s1_body starts with "Step 1:", so we strip that prefix.
                    s1_tail = s1_body[len("Step 1:"):]
                    ctx_text = prompt + s1_tail + "\nStep 2:"

                pairs.append({
                    "problem": problem,
                    "ctx_text": ctx_text,
                    "pos_tail": pos_tail,
                    "neg_tail": neg_tail,
                    "context_idx": k,
                })

    print(f"Generated {len(pairs)} pairs pre-length-filter")
    print(f"  skipped (no negatives): {skipped_no_neg}")
    print(f"  skipped (parse fail): {sanity_fail}")

    # Tokenize r+ and r- for length-filtering
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(REPO_ROOT, "checkpoints/sft_24_tot_merged"),
        trust_remote_code=True,
    )

    filtered = []
    dropped_len = 0
    for p in pairs:
        pos_ids = tokenizer.encode(p["pos_tail"], add_special_tokens=False)
        neg_ids = tokenizer.encode(p["neg_tail"], add_special_tokens=False)
        if abs(len(pos_ids) - len(neg_ids)) > LEN_DIFF_CAP:
            dropped_len += 1
            continue
        p["pos_len"] = len(pos_ids)
        p["neg_len"] = len(neg_ids)
        filtered.append(p)

    print(f"After length-filter (|Δ|≤{LEN_DIFF_CAP}): {len(filtered)} pairs "
          f"(dropped {dropped_len})")

    # Write
    with open(OUTPUT, "w") as f:
        for p in filtered:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote to {OUTPUT}")

    # Stats
    ctx0 = sum(1 for p in filtered if p["context_idx"] == 0)
    ctx1 = sum(1 for p in filtered if p["context_idx"] == 1)
    unique_probs = len(set(p["problem"] for p in filtered))
    print(f"Breakdown: ctx0={ctx0}, ctx1={ctx1}, unique_problems={unique_probs}")


if __name__ == "__main__":
    main()
