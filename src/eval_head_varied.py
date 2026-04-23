"""Quick origin-distance eval for the varied-G24 head.

For each val/test tree in the sampled cache:
  - Forward sampled hidden states through the head.
  - Compute d_hyp(0, z_i) per node.
  - Compare against v_values[i] (BFS distance to nearest success terminal).

Reports Spearman rank correlation (main metric for origin_ranking loss),
mean d_origin for success vs non-success nodes, and percentage of pairs
(i, j) with v[i] < v[j] where d_hyp(0, z_i) < d_hyp(0, z_j) (the ranking
accuracy the loss directly targets).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.head import HyperbolicHead
from src.train_head import TreeCacheDataset


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def eval_split(head: HyperbolicHead, ds: TreeCacheDataset, device,
               max_trees: int | None = None,
               rank_pairs_per_tree: int = 500) -> dict:
    d_o_all: list[np.ndarray] = []
    v_all: list[np.ndarray] = []
    d_o_success: list[float] = []
    d_o_nonsuccess: list[float] = []
    rank_correct = 0
    rank_total = 0

    rng = np.random.default_rng(0)
    n_trees = len(ds) if max_trees is None else min(max_trees, len(ds))

    for idx in range(n_trees):
        item = ds[idx]
        n = item["n"]
        if n < 2 or "v_values" not in item:
            continue
        v = np.asarray(item["v_values"])
        # Drop unreachable
        valid = np.where(v >= 0)[0]
        if len(valid) < 2:
            continue

        hidden = torch.as_tensor(item["hidden"][valid], device=device,
                                 dtype=torch.float32)
        z = head(hidden)
        d_o = head.origin_distance(z).float().cpu().numpy()
        v_val = v[valid]
        is_succ = np.asarray(item["is_success"])[valid]

        d_o_all.append(d_o)
        v_all.append(v_val)
        d_o_success.extend(d_o[is_succ].tolist())
        d_o_nonsuccess.extend(d_o[~is_succ].tolist())

        # Sample pairs with v[i] != v[j]; count rank-correct
        if len(valid) >= 2:
            k = min(rank_pairs_per_tree, len(valid) * (len(valid) - 1) // 2)
            i = rng.integers(0, len(valid), size=k)
            j = rng.integers(0, len(valid), size=k)
            mask = v_val[i] != v_val[j]
            if mask.any():
                i, j = i[mask], j[mask]
                # Orient so v[i] < v[j]
                flip = v_val[i] > v_val[j]
                ii = np.where(flip, j, i)
                jj = np.where(flip, i, j)
                # Correct if d_o[ii] < d_o[jj]
                rank_correct += int((d_o[ii] < d_o[jj]).sum())
                rank_total += len(ii)

    d_o_cat = np.concatenate(d_o_all) if d_o_all else np.zeros(0)
    v_cat = np.concatenate(v_all) if v_all else np.zeros(0)
    return {
        "n_trees": n_trees,
        "n_nodes": int(len(d_o_cat)),
        "spearman_d_origin_vs_v": spearman_rho(d_o_cat, v_cat) if len(d_o_cat) else 0.0,
        "mean_d_origin_success": float(np.mean(d_o_success)) if d_o_success else 0.0,
        "mean_d_origin_nonsuccess": float(np.mean(d_o_nonsuccess)) if d_o_nonsuccess else 0.0,
        "n_success": len(d_o_success),
        "n_nonsuccess": len(d_o_nonsuccess),
        "rank_accuracy": (rank_correct / rank_total) if rank_total else 0.0,
        "n_rank_pairs": rank_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_ckpt", required=True)
    ap.add_argument("--tree_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max_trees", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.head_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    head = HyperbolicHead(
        in_dim=ckpt["in_dim"],
        hyp_dim=cfg["model"]["hyp_dim"],
        hidden_dims=cfg["model"]["head_hidden_dims"],
        manifold=cfg["model"]["manifold"],
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    results: dict = {}
    for split in args.splits:
        ds = TreeCacheDataset(args.tree_dir, split)
        r = eval_split(head, ds, device, max_trees=args.max_trees)
        print(f"\n=== {split} (max_trees={args.max_trees}) ===")
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        results[split] = r

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
