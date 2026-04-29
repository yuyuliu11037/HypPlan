"""Generate N-Queens train/val/test JSONL with universe-partition split.

Design:
  - For each N in --train_Ns (default 5,6,7,8) and each k in --ks
    (default 0,1,2,3,4 capped at N-1), enumerate the full universe of
    distinct length-k prefixes that extend to a valid full solution.
  - For N == --test_N (default 8) and each k, randomly partition the
    universe into a TEST split of size --n_test_per_k (or all if smaller)
    and a TRAIN split (the rest). The test seed is fixed by --seed for
    reproducibility.
  - For N != --test_N, all universe members go to train.
  - Train + val are sampled from the train pool with disjoint indices.

Schema per record:
  {
    "id": str,
    "N": int,
    "k": int,
    "prefix": list[int],            # length k, 1-indexed columns
    "gold_extension": list[int],    # one valid full solution (lex-min)
  }

Train/test disjointness invariant: NO (N, k, prefix) tuple appears in both.

Usage:
  python3.10 data/generate_data_nqueens_full.py \\
    --train_Ns 5,6,7,8 --test_N 8 --ks 0,1,2,3,4 \\
    --n_test_per_k 12 --val_frac 0.05 --seed 12345 \\
    --out_train data/nqueens_train.jsonl \\
    --out_val   data/nqueens_val.jsonl \\
    --out_test  data/nqueens_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_nqueens import (all_distinct_prefixes, solve_lex_min)


def _gold_extension(N: int, prefix: list[int]) -> list[int] | None:
    if not prefix:
        return solve_lex_min(N)
    placed = [(r, c) for r, c in enumerate(prefix, 1)]
    return solve_lex_min(N, placed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_Ns", default="5,6,7,8")
    ap.add_argument("--test_N", type=int, default=8)
    ap.add_argument("--ks", default="0,1,2,3,4")
    ap.add_argument("--n_test_per_k", type=int, default=12,
                    help="problems per k held out for test at test_N")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="fraction of TRAIN pool reserved for val")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_train", default="data/nqueens_train.jsonl")
    ap.add_argument("--out_val", default="data/nqueens_val.jsonl")
    ap.add_argument("--out_test", default="data/nqueens_test.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_Ns = [int(x) for x in args.train_Ns.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    # Step 1: enumerate universe for every (N, k) pair.
    universe: dict[tuple[int, int], list[list[int]]] = {}
    for N in train_Ns:
        for k in ks:
            if k >= N:
                continue
            prefs = all_distinct_prefixes(N, k)
            universe[(N, k)] = prefs
            print(f"  N={N} k={k}: {len(prefs)} prefixes", flush=True)

    # Step 2: partition test_N universe into test/train splits
    test_records: list[dict] = []
    train_pool: list[dict] = []

    for (N, k), prefs in universe.items():
        # Stable shuffle per (N, k) so the split is reproducible.
        local_rng = random.Random(args.seed + 1000 * N + k)
        shuffled = list(prefs)
        local_rng.shuffle(shuffled)

        if N == args.test_N:
            n_test = min(args.n_test_per_k, len(shuffled))
            test_part = shuffled[:n_test]
            train_part = shuffled[n_test:]
            for p in test_part:
                gold = _gold_extension(N, p)
                test_records.append({
                    "N": N, "k": k, "prefix": list(p),
                    "gold_extension": list(gold) if gold else None,
                })
            for p in train_part:
                gold = _gold_extension(N, p)
                train_pool.append({
                    "N": N, "k": k, "prefix": list(p),
                    "gold_extension": list(gold) if gold else None,
                })
        else:
            for p in shuffled:
                gold = _gold_extension(N, p)
                train_pool.append({
                    "N": N, "k": k, "prefix": list(p),
                    "gold_extension": list(gold) if gold else None,
                })

    # Step 3: split train_pool into train + val.
    rng.shuffle(train_pool)
    n_val = max(1, int(len(train_pool) * args.val_frac))
    val_records = train_pool[:n_val]
    train_records = train_pool[n_val:]

    # Step 4: assign IDs and write.
    def write(records, out_path, prefix_label):
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            for i, r in enumerate(records):
                r["id"] = f"nqueens_{prefix_label}_{i:05d}"
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(records):>5d} -> {out_p}", flush=True)

    write(train_records, args.out_train, "train")
    write(val_records, args.out_val, "val")
    write(test_records, args.out_test, "test")

    # Step 5: invariants.
    train_keys = {(r["N"], r["k"], tuple(r["prefix"])) for r in train_records}
    val_keys = {(r["N"], r["k"], tuple(r["prefix"])) for r in val_records}
    test_keys = {(r["N"], r["k"], tuple(r["prefix"])) for r in test_records}
    overlap_tt = train_keys & test_keys
    overlap_tv = train_keys & val_keys
    overlap_vt = val_keys & test_keys
    print(f"\n  train ∩ test : {len(overlap_tt)}  (must be 0)")
    print(f"  train ∩ val  : {len(overlap_tv)}  (must be 0)")
    print(f"  val   ∩ test : {len(overlap_vt)}  (must be 0)")
    assert not overlap_tt, "train/test overlap"
    assert not overlap_tv, "train/val overlap"
    assert not overlap_vt, "val/test overlap"

    # Per-N breakdown.
    from collections import Counter
    print(f"\n  Train N-distribution: "
          f"{Counter(r['N'] for r in train_records)}")
    print(f"  Test  N-distribution: "
          f"{Counter(r['N'] for r in test_records)}")
    print(f"  Test  k-distribution: "
          f"{Counter(r['k'] for r in test_records)}")
    print("\nUniverse partition done.", flush=True)


if __name__ == "__main__":
    main()
