"""Generate CLUTRR-like JSONL data for Group B (kinship reasoning OOD probe).

Difficulty knob: hop count `k`. OOD eval defaults: k in {2,3,4}, evenly
distributed across the test set so the same eval has an internal
difficulty gradient.

Schema per line:
  {
    "id": str,
    "k": int,                                    # hop count
    "entities": list[str],                        # length k+1
    "edges": list[[i, relation, j]],              # k facts, shuffled order
    "query": [head_idx, tail_idx],
    "answer": str,                                # composed kinship term
    "chain": list[str],                           # ordered chain relations
    "prompt": str,                                # full NL story + question
    "init_state_text": str,                       # for Stage-1 head input
    "answer_label": str,                          # gold trajectory
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_clutrr import (
    Problem, format_gold_trajectory, generate_problem, render_state,
)


def problem_to_record(p: Problem, idx: int, split: str) -> dict:
    return {
        "id": f"clutrr_{split}_{idx}",
        "k": len(p.chain),
        "entities": list(p.entities),
        "edges": [[i, rel, j] for (i, rel, j) in p.edges],
        "query": [p.query[0], p.query[1]],
        "answer": p.answer,
        "chain": list(p.chain),
        "prompt": p.render_problem(),
        "init_state_text": p.render_problem(),
        "answer_label": format_gold_trajectory(p),
        "split": split,
    }


def gen_split(n: int, ks: list[int], seed_offset: int, split: str) -> list[dict]:
    out: list[dict] = []
    di = 0
    seed = seed_offset
    while len(out) < n:
        k = ks[di % len(ks)]
        di += 1
        try:
            p = generate_problem(k=k, seed=seed)
            out.append(problem_to_record(p, len(out), split))
        except RuntimeError:
            pass
        seed += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--ks", default="2,3,4")
    ap.add_argument("--seed_base", type=int, default=9101)
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
        print(f"Generating clutrr/{split}: n={n}, k in {ks}", flush=True)
        recs = gen_split(n=n, ks=ks, seed_offset=seed, split=split)
        path = out_dir / f"clutrr_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
