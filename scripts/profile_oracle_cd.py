"""Profile CountdownOracle timing on cached problems.

Measures per-problem wall time for:
  - can_reach(root_pool): cold cache, fills the memo as a side effect
  - winning_ops(root_pool): reuses the filled memo, so this is the "hot" number
  - winning_ops on an interior state (after one random legal op): what DAgger
    actually queries at step-2+ boundaries

DAgger calls winning_ops at every step boundary on the states reached by the
policy. A full 3-epoch DAgger run with 1000 train × 3 rollouts × ~5 boundaries
= ~45k queries. p95 * 45000 gives a rough total-budget estimate.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.oracle_cd import CountdownOracle, apply_step


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def profile(jsonl_path: Path, n_profile: int, seed: int) -> None:
    with jsonl_path.open() as f:
        problems = [json.loads(line) for line in f]
    if n_profile > 0:
        problems = problems[:n_profile]

    t_can_reach: list[float] = []
    t_winning_root: list[float] = []
    t_winning_interior: list[float] = []
    winners_per_root: list[int] = []
    winners_per_interior: list[int] = []
    rng = random.Random(seed)
    total_wall = 0.0

    t_total = time.time()
    for p in problems:
        pool = tuple(sorted(p["pool"]))
        target = p["target"]
        oracle = CountdownOracle(target)

        t0 = time.time()
        reachable = oracle.can_reach(pool)
        t_can_reach.append(time.time() - t0)
        assert reachable, f"unsolvable problem in cache: {p}"

        t0 = time.time()
        root_winners = oracle.winning_ops(pool)
        t_winning_root.append(time.time() - t0)
        winners_per_root.append(len(root_winners))

        # Apply one winning op to reach an interior state (size 5)
        if root_winners:
            sym, a, b, r = rng.choice(root_winners)
            interior_pool = apply_step(pool, a, b, r)
            t0 = time.time()
            interior_winners = oracle.winning_ops(interior_pool)
            t_winning_interior.append(time.time() - t0)
            winners_per_interior.append(len(interior_winners))

    total_wall = time.time() - t_total

    def _fmt(xs: list[float], label: str) -> str:
        if not xs:
            return f"  {label}: n/a"
        return (f"  {label:24s} mean={statistics.mean(xs)*1000:7.2f}ms  "
                f"p50={statistics.median(xs)*1000:7.2f}ms  "
                f"p95={_pct(xs, 0.95)*1000:7.2f}ms  "
                f"max={max(xs)*1000:7.2f}ms")

    print(f"\nProfiled {len(problems)} problems from {jsonl_path}")
    print(f"total wall: {total_wall:.2f}s  avg per-problem: "
          f"{total_wall/len(problems)*1000:.1f}ms")
    print(_fmt(t_can_reach, "can_reach(root, cold)"))
    print(_fmt(t_winning_root, "winning_ops(root, hot)"))
    print(_fmt(t_winning_interior, "winning_ops(size-5 interior)"))
    if winners_per_root:
        print(f"  avg winners at root:     "
              f"{statistics.mean(winners_per_root):5.1f}  "
              f"(min={min(winners_per_root)}, max={max(winners_per_root)})")
    if winners_per_interior:
        print(f"  avg winners at size-5:   "
              f"{statistics.mean(winners_per_interior):5.1f}  "
              f"(min={min(winners_per_interior)}, max={max(winners_per_interior)})")

    # Budget extrapolation
    p95_hot = _pct(t_winning_root, 0.95)
    est_queries = 1000 * 3 * 5 * 3  # train × rollouts × boundaries × epochs
    print(f"\nBudget estimate for 3-epoch DAgger (1000 train × 3 rollouts × "
          f"~5 boundaries):\n"
          f"  p95 winning_ops (hot) × {est_queries:,} queries = "
          f"{p95_hot * est_queries / 60:.1f} min "
          f"(assuming each problem gets a fresh oracle instance)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/cd_val.jsonl")
    ap.add_argument("--n", type=int, default=0,
                    help="limit to first N problems (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    profile(Path(args.data), args.n, args.seed)
