"""Generate solvable Countdown problems.

Pool = 6 numbers: 5 small (uniform 1..10 with replacement) + 1 big (25/50/75/100).
Target = uniform 100..999. Filter for solvability via CountdownOracle.

Writes JSONL with one problem per line: {"pool": [int,...], "target": int,
"problem_idx": int}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.oracle_cd import CountdownOracle


BIG_NUMBERS = [25, 50, 75, 100]


def sample_problem(rng: random.Random) -> tuple[list[int], int]:
    small = [rng.randint(1, 10) for _ in range(5)]
    big = rng.choice(BIG_NUMBERS)
    pool = sorted(small + [big])
    target = rng.randint(100, 999)
    return pool, target


def generate_split(n_problems: int, seed: int, out_path: Path,
                   raw_budget_per_kept: int = 20) -> None:
    rng = random.Random(seed)
    problems: list[dict] = []
    raw = 0
    t0 = time.time()
    max_raw = n_problems * raw_budget_per_kept
    while len(problems) < n_problems and raw < max_raw:
        raw += 1
        pool, target = sample_problem(rng)
        oracle = CountdownOracle(target)
        if oracle.can_reach(tuple(pool)):
            problems.append({"pool": pool, "target": target,
                             "problem_idx": len(problems)})
        if raw % 100 == 0:
            elapsed = time.time() - t0
            rate = len(problems) / raw if raw else 0
            print(f"  [{elapsed:.1f}s] raw={raw} kept={len(problems)} "
                  f"solvability={rate:.2%}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    solvability = len(problems) / raw if raw else 0
    print(f"  Wrote {len(problems)} to {out_path} in {time.time()-t0:.1f}s "
          f"(solvability {solvability:.2%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Generating Countdown splits (N=6, pool=5×small+1×big, target∈[100,999])")
    for split, n, seed_offset in [("train", args.train, 0),
                                   ("val", args.val, 1),
                                   ("test", args.test, 2)]:
        print(f"[{split}]")
        generate_split(n, args.seed + seed_offset,
                       out_dir / f"cd_{split}.jsonl")


if __name__ == "__main__":
    main()
