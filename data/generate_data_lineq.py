"""Generate single-variable linear-equation JSONL data (Group A OOD #1).

Difficulty knob `k`: number of canonical solving steps. OOD eval uses
k ∈ {3,4,5} (k=3 baseline, k=4 multi-term-x, k=5 multi-term-x and -const).

Schema per line:
  {
    "id": str,
    "k": int,
    "initial": {"lhs_x": [...], "lhs_c": [...], "rhs_x": [...], "rhs_c": [...]},
    "solution": int,
    "prompt": str,                 # NL question
    "init_state_text": str,
    "answer_label": str,           # gold trajectory
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_lineq import (
    Problem, format_gold_trajectory, format_question, generate_problem,
    render_state,
)


def problem_to_record(p: Problem, idx: int, split: str, k: int) -> dict:
    return {
        "id": f"lineq_{split}_{idx}",
        "k": k,
        "initial": {
            "lhs_x": list(p.initial.lhs_x),
            "lhs_c": list(p.initial.lhs_c),
            "rhs_x": list(p.initial.rhs_x),
            "rhs_c": list(p.initial.rhs_c),
        },
        "solution": int(p.solution),
        "prompt": format_question(p),
        "init_state_text": render_state(p, p.initial),
        "answer_label": format_gold_trajectory(p),
        "split": split,
    }


def gen_split(n: int, ks: list[int], seed_offset: int, split: str) -> list[dict]:
    out: list[dict] = []
    di = 0
    seed = seed_offset
    fail_streak = 0
    while len(out) < n and fail_streak < 1000:
        k = ks[di % len(ks)]
        di += 1
        try:
            p = generate_problem(k=k, seed=seed)
            out.append(problem_to_record(p, len(out), split, k))
            fail_streak = 0
        except RuntimeError:
            fail_streak += 1
        seed += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--ks", default="3,4,5")
    ap.add_argument("--seed_base", type=int, default=7777)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.n_train, args.seed_base),
        ("val", args.n_val, args.seed_base + 100_000),
        ("test", args.n_test, args.seed_base + 200_000),
    ]
    for split, n, seed in splits:
        if n <= 0:
            continue
        print(f"Generating lineq/{split}: n={n}, k in {ks}", flush=True)
        recs = gen_split(n=n, ks=ks, seed_offset=seed, split=split)
        path = out_dir / f"lineq_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
