"""Recompute v_values in saved Countdown tree files under the continuous v
definition (v(s) = smallest |final_value - target| achievable from s).

Re-runs `sample_tree` deterministically with the same seed that was used in
`data/generate_tree_data_cd.py`, extracts v_value per node, and overwrites
only the `v_values` field of each saved `problem_{idx}.pt` — hidden states
(`hidden_{idx}.npy`) are untouched because the tree topology is unchanged.

Fast (CPU only, no LLM forward). Full train+val+test takes ~1-2 min.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tree_data_cd import sample_tree


def recompute_split(data_dir: Path, cache_dir: Path, trees_dir: Path,
                    split: str, n_trajectories: int, n_guided: int) -> None:
    jsonl = data_dir / f"cd_{split}.jsonl"
    cache_file = cache_dir / f"{split}.pkl"
    tree_split_dir = trees_dir / split
    if not (jsonl.exists() and cache_file.exists() and tree_split_dir.exists()):
        print(f"[{split}] skip — missing inputs")
        return

    with jsonl.open() as f:
        problems = [json.loads(line) for line in f]
    with cache_file.open("rb") as f:
        caches = pickle.load(f)

    t0 = time.time()
    updated = 0
    skipped = 0
    mismatched = 0
    for idx, (p, memo) in enumerate(zip(problems, caches)):
        meta_path = tree_split_dir / f"problem_{idx}.pt"
        if not meta_path.exists():
            skipped += 1
            continue
        meta = torch.load(meta_path, weights_only=False)

        tree = sample_tree(p["pool"], p["target"], memo,
                           n_trajectories=n_trajectories,
                           n_guided=n_guided,
                           seed=p["problem_idx"])
        # Sanity check: regenerated tree topology must match the saved meta
        if tree.n != meta["n"]:
            mismatched += 1
            print(f"  [{split}] idx={idx}: tree size mismatch "
                  f"(new={tree.n}, saved={meta['n']}) — skipping")
            continue

        new_v = np.array([n.v_value for n in tree.nodes], dtype=np.int32)
        meta["v_values"] = new_v
        torch.save(meta, meta_path)
        updated += 1

    print(f"[{split}] updated={updated} skipped={skipped} "
          f"mismatched={mismatched} in {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--cache_dir", type=str, default="data/cd_oracle_cache")
    ap.add_argument("--trees_dir", type=str, default="data/cd_trees")
    ap.add_argument("--n_trajectories", type=int, default=300)
    ap.add_argument("--n_guided", type=int, default=100)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    trees_dir = Path(args.trees_dir)

    for split in ["train", "val", "test"]:
        recompute_split(data_dir, cache_dir, trees_dir, split,
                        args.n_trajectories, args.n_guided)


if __name__ == "__main__":
    main()
