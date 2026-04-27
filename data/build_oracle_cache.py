"""Build offline oracle caches for Countdown.

For each problem in a split, warm up a CountdownOracle memo by calling
can_reach on the root pool (which transitively fills the memo with every
reachable descendant state). Serialize the per-problem dict
{sorted_pool_tuple: bool} to disk.

At DAgger rollout time, load `caches[problem_idx]` into a fresh oracle's
_cache — any state the policy legally reaches is an O(1) lookup.

Output: one pickle per split, each a list[dict] parallel to the JSONL.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.oracle_cd import CountdownOracle


def build(jsonl_path: Path, out_path: Path) -> None:
    with jsonl_path.open() as f:
        problems = [json.loads(line) for line in f]

    caches: list[dict] = []
    t0 = time.time()
    total_states = 0
    for i, p in enumerate(problems):
        pool = tuple(sorted(p["pool"]))
        oracle = CountdownOracle(p["target"])
        oracle.can_reach(pool)
        caches.append(dict(oracle._cache))
        total_states += len(oracle._cache)
        if (i + 1) % 100 == 0:
            print(f"  [{time.time()-t0:.1f}s] built {i+1}/{len(problems)} "
                  f"avg states/problem={total_states/(i+1):.0f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(caches, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {out_path} ({size_mb:.1f} MB, "
          f"{total_states:,} states across {len(problems)} problems)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="data/cd_oracle_cache")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    for split in ["train", "val", "test"]:
        jsonl = data_dir / f"cd_{split}.jsonl"
        if not jsonl.exists():
            print(f"[{split}] skip — {jsonl} missing")
            continue
        print(f"[{split}]")
        build(jsonl, out_dir / f"{split}.pkl")


if __name__ == "__main__":
    main()
