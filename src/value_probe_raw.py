"""Baseline for value_probe: regress value on RAW LLM hidden states (no head).

Tests whether value information is *present* in the frozen LLM's hidden state
before the head touches it. If raw-hidden R^2 >> head R^2, the head is
throwing away value information. If raw-hidden R^2 is also small, the LLM
never encoded it in the first place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.train_head import TreeCacheDataset
from src.value_probe import compute_node_values, linear_probe_r2, pearson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--tree_dir", default="data/trees")
    ap.add_argument("--max_trees", type=int, default=None)
    ap.add_argument("--nodes_cap", type=int, default=200_000,
                    help="Cap on total nodes (raw hidden is 4096-dim; 200k * 4096 = 3.3GB).")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ds = TreeCacheDataset(args.tree_dir, args.split)
    n_trees = len(ds) if args.max_trees is None else min(args.max_trees, len(ds))
    print(f"split={args.split} n_trees={n_trees}")

    Hs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    LTs: list[np.ndarray] = []

    rng = np.random.default_rng(0)
    total = 0
    for ti in range(n_trees):
        item = ds[ti]
        n = item["n"]
        if n < 2:
            continue
        meta = torch.load(ds.split_dir / f"problem_{ds.indices[ti]}.pt",
                          weights_only=False)
        val, lt, _ = compute_node_values(item["parents"], item["is_terminal"],
                                         meta["is_success"])
        hidden = np.asarray(item["hidden"], dtype=np.float32)
        Hs.append(hidden)
        Ys.append(val)
        LTs.append(lt)
        total += n
        if total >= args.nodes_cap:
            break

    H = np.concatenate(Hs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    LT = np.concatenate(LTs, axis=0)
    mask = np.isfinite(Y)
    H = H[mask]; Y = Y[mask]; LT = LT[mask]

    # Random subsample to nodes_cap (otherwise 4096-wide closed-form solve explodes)
    if H.shape[0] > args.nodes_cap:
        pick = rng.choice(H.shape[0], size=args.nodes_cap, replace=False)
        H = H[pick]; Y = Y[pick]; LT = LT[pick]

    print(f"using {len(Y)} nodes ({(LT > 1).sum()} non-leaf)")

    probe_all = linear_probe_r2(H, Y.astype(np.float64), ridge=1.0)
    nonleaf = LT > 1
    if nonleaf.sum() > 100:
        probe_nonleaf = linear_probe_r2(H[nonleaf], Y[nonleaf].astype(np.float64),
                                        ridge=1.0)
    else:
        probe_nonleaf = None

    result = {
        "split": args.split,
        "n_nodes": int(len(Y)),
        "n_nonleaf": int(nonleaf.sum()),
        "raw_hidden_probe_all": probe_all,
        "raw_hidden_probe_nonleaf": probe_nonleaf,
    }
    print(json.dumps(result, indent=2))

    out = args.output or f"results/value_probe/raw_hidden_{args.split}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
