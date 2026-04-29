"""Generate N-Queens test JSONL with mixed-k pre-placed queens.

Each record specifies a (N, k, prefix) initial state where prefix is a
length-k column-per-row vector (1-indexed) of queens already placed.
The model's task is to extend the prefix to a full valid placement.

Schema:
  {
    "id": str,
    "N": int,
    "k": int,
    "prefix": list[int],            # length k, 1-indexed columns for rows 1..k
    "gold_extension": list[int]      # one valid full solution starting w/ prefix
  }

We sample uniformly without replacement from the union of all distinct
(k, prefix) tuples for k in --ks. This naturally biases toward k values
with larger universe (k=2,3 in N=8 cover most of the 107-tuple space).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_nqueens import (all_distinct_prefixes,
                                  solve_lex_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="data/nqueens_test.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ks = [int(x) for x in args.ks.split(",")]

    universe: list[tuple[int, list[int]]] = []
    per_k_count: dict[int, int] = {}
    for k in ks:
        prefs = all_distinct_prefixes(args.N, k)
        per_k_count[k] = len(prefs)
        for p in prefs:
            universe.append((k, p))
        print(f"N={args.N} k={k}: {len(prefs)} distinct prefixes",
                flush=True)
    print(f"Total universe: {len(universe)}", flush=True)

    if args.n_test > len(universe):
        raise SystemExit(
            f"n_test={args.n_test} exceeds universe size {len(universe)}")

    rng.shuffle(universe)
    chosen = universe[:args.n_test]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = {k: 0 for k in ks}
    with open(out_path, "w") as f:
        for i, (k, prefix) in enumerate(chosen):
            counts[k] += 1
            if k == 0:
                gold = solve_lex_min(args.N)
            else:
                placed = [(r, c) for r, c in enumerate(prefix, 1)]
                gold = solve_lex_min(args.N, placed)
            rec = {
                "id": f"nqueens_N{args.N}_{i:03d}",
                "N": args.N,
                "k": k,
                "prefix": prefix,
                "gold_extension": gold,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(chosen)} records -> {out_path}", flush=True)
    print("Distribution: " +
            ", ".join(f"k={k}:{counts[k]}" for k in ks), flush=True)


if __name__ == "__main__":
    main()
