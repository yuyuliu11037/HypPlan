"""Balanced variant of generate_24_varied.py.

Differences from the original:
- Ensures target=24 makes up >= 40% of each split.
- Filters "lazy" targets: drops records where target is already an element of
  pool (trivially "do nothing useful") or target < 10 (small targets are hit
  by many simple ops, teaching little).
- target=24 is NOT deduped across source problems: the model sees more
  (pool, 24) variation grounded in different parent problems.

Output schema same as before:
  {"pool": [int, ...], "target": int, "n_steps": int,
   "source_problem": "a,b,c,d", "split": "train|val|test"}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.varied_24 import iter_varied_pairs


def unique_problems(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with path.open() as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def is_lazy(pair: dict) -> bool:
    """Drop trivially solvable cases that don't teach hard reasoning."""
    target = pair["target"]
    pool = pair["pool"]
    if target < 10:
        return True
    if target in pool:
        return True
    return False


def collect_pairs_by_target(problem: str
                              ) -> tuple[list[dict], list[dict]]:
    """Returns (target24_pairs, varied_pairs_filtered) for one problem.

    target24_pairs: every (pool, 24) reachable in this tree (history-distinct
        but pool-distinct dedup, since same pool same target is identical).
    varied_pairs_filtered: every (pool, target!=24) after dropping lazy cases.
    """
    seen_24: set = set()
    target24: list[dict] = []
    varied_seen: set = set()
    varied: list[dict] = []
    for pair in iter_varied_pairs(problem):
        if pair["target"] == 24:
            key = tuple(pair["pool"])
            if key in seen_24:
                continue
            seen_24.add(key)
            target24.append(pair)
        else:
            if is_lazy(pair):
                continue
            key = (tuple(pair["pool"]), pair["target"])
            if key in varied_seen:
                continue
            varied_seen.add(key)
            varied.append(pair)
    return target24, varied


def build_split(problems: list[str], target_24_min_frac: float,
                cap_total: int, rng: random.Random) -> list[dict]:
    """Aggregate per-problem pairs and balance to >= target_24_min_frac."""
    all_24: list[dict] = []
    all_var: list[dict] = []
    for p in problems:
        t24, var = collect_pairs_by_target(p)
        all_24.extend(t24)
        all_var.extend(var)

    rng.shuffle(all_24)
    rng.shuffle(all_var)

    # We want: n_24 / (n_24 + n_var) >= target_24_min_frac
    # And total <= cap_total.
    n_24_target = int(cap_total * target_24_min_frac)
    n_var_target = cap_total - n_24_target
    n_24 = min(len(all_24), n_24_target)
    n_var = min(len(all_var), n_var_target)

    # If we have fewer var than target, use all of them.
    # If n_24 < n_24_target (we have fewer target=24 than we wanted), keep
    # the ratio by capping n_var.
    if n_24 < n_24_target:
        n_var = min(n_var, int(n_24 * (1 - target_24_min_frac) / target_24_min_frac))

    sample = all_24[:n_24] + all_var[:n_var]
    rng.shuffle(sample)
    return sample, len(all_24), len(all_var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_prefix", default="24_varied_bal")
    ap.add_argument("--target_24_frac", type=float, default=0.4)
    ap.add_argument("--cap_train", type=int, default=6000)
    ap.add_argument("--cap_val", type=int, default=600)
    ap.add_argument("--cap_test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)

    for split, cap in [("train", args.cap_train),
                        ("val", args.cap_val),
                        ("test", args.cap_test)]:
        in_path = data_dir / f"24_{split}.jsonl"
        problems = unique_problems(in_path)
        print(f"[{split}] {len(problems)} unique problems", flush=True)
        records, n24_avail, nvar_avail = build_split(
            problems, args.target_24_frac, cap, rng)
        # Tag with split
        for r in records:
            r["split"] = split

        out_path = data_dir / f"{args.out_prefix}_{split}.jsonl"
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        # Stats
        n24 = sum(1 for r in records if r["target"] == 24)
        ns = Counter(r["n_steps"] for r in records)
        targets = Counter(r["target"] for r in records)
        top = targets.most_common(5)
        print(f"  available: target=24 {n24_avail}, varied {nvar_avail}",
              flush=True)
        print(f"  written: {len(records)} -> target=24 {n24} ({n24/len(records):.2%}), "
              f"unique targets {len(targets)}", flush=True)
        print(f"  n_steps dist: {dict(ns)}", flush=True)
        print(f"  top targets: {top}", flush=True)


if __name__ == "__main__":
    main()
