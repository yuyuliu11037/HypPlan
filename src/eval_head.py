"""Stage-1 evaluation: distortion + rank metrics + qualitative tree viz.

Loads a trained head checkpoint, iterates over val and test tree caches, and
reports:
  - mean |d_H - d_tree|
  - mean relative distortion (|d_H - d_tree| / d_tree) on non-root pairs
  - Spearman rank correlation between d_H and d_tree (pooled over pairs)
  - d_tree vs d_hyp scatter (png)
  - 2D tangent-PCA tree plots for a few demo problems
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from src.head import HyperbolicHead
from src.tree_data import pair_distances_lca
from src.train_head import TreeCacheDataset, sample_pairs


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Uses scipy if available, else manual."""
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y, nan_policy="omit").correlation)
    except Exception:
        # Manual: average ranks, Pearson on ranks
        def rankdata(a):
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(a) + 1)
            return ranks
        rx = rankdata(x)
        ry = rankdata(y)
        rx -= rx.mean(); ry -= ry.mean()
        denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / denom) if denom > 0 else 0.0


@torch.no_grad()
def evaluate_split(head: HyperbolicHead, ds: TreeCacheDataset, device,
                    pairs_per_tree: int, max_trees: int | None = None) -> dict:
    d_tree_all: list[np.ndarray] = []
    d_hyp_all: list[np.ndarray] = []
    abs_err = 0.0
    rel_err = 0.0
    rel_count = 0
    count = 0
    n_trees = len(ds) if max_trees is None else min(max_trees, len(ds))
    import random
    rng = random.Random(0)
    for idx in range(n_trees):
        item = ds[idx]
        n = item["n"]
        if n < 2:
            continue
        hidden = torch.as_tensor(item["hidden"], device=device, dtype=torch.float32)
        z = head(hidden)

        pairs = sample_pairs(n, pairs_per_tree, rng)
        if pairs.shape[0] == 0:
            continue
        dt = pair_distances_lca(item["parents"], item["depths"],
                                 pairs[:, 0], pairs[:, 1]).astype(np.float32)
        i = torch.as_tensor(pairs[:, 0], device=device, dtype=torch.long)
        j = torch.as_tensor(pairs[:, 1], device=device, dtype=torch.long)
        dh = head.distance(z[i], z[j]).float().cpu().numpy()

        d_tree_all.append(dt)
        d_hyp_all.append(dh)
        err = np.abs(dh - dt)
        abs_err += err.sum()
        count += err.size
        mask = dt > 0
        if mask.any():
            rel_err += (err[mask] / dt[mask]).sum()
            rel_count += int(mask.sum())

    dt_cat = np.concatenate(d_tree_all) if d_tree_all else np.zeros(0)
    dh_cat = np.concatenate(d_hyp_all) if d_hyp_all else np.zeros(0)
    return {
        "n_pairs": int(count),
        "n_trees": n_trees,
        "mean_abs_distortion": float(abs_err / max(count, 1)),
        "mean_rel_distortion": float(rel_err / max(rel_count, 1)),
        "spearman": spearman(dh_cat, dt_cat) if len(dt_cat) > 0 else 0.0,
        "d_tree_flat": dt_cat,
        "d_hyp_flat": dh_cat,
    }


def save_scatter(d_tree: np.ndarray, d_hyp: np.ndarray, path: Path,
                  max_points: int = 10000):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"matplotlib unavailable; skipping {path}")
        return
    n = len(d_tree)
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        d_tree = d_tree[sel]
        d_hyp = d_hyp[sel]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d_tree + np.random.uniform(-0.1, 0.1, size=len(d_tree)),
               d_hyp, s=2, alpha=0.3)
    m = max(d_tree.max() if len(d_tree) else 1, d_hyp.max() if len(d_hyp) else 1)
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("tree distance (edges)")
    ax.set_ylabel("hyperbolic distance")
    ax.set_title(f"d_tree vs d_hyp (n={len(d_tree)})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def visualize_tree_2d(head: HyperbolicHead, ds: TreeCacheDataset,
                        tree_idx: int, device, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    item = ds[tree_idx]
    n = item["n"]
    hidden = torch.as_tensor(item["hidden"], device=device, dtype=torch.float32)
    z = head(hidden)                       # (n, hyp_dim) or (n, hyp_dim+1)

    # Take the tangent-space vectors out of the head (pre-exp-map) for a
    # principled 2D projection via PCA on tangent vectors.
    v = head.mlp(hidden)                   # (n, hyp_dim)
    v_np = v.cpu().numpy()
    v_mean = v_np.mean(axis=0, keepdims=True)
    vc = v_np - v_mean
    # SVD of vc
    try:
        U, S, Vt = np.linalg.svd(vc, full_matrices=False)
        v2 = vc @ Vt[:2].T                 # (n, 2)
    except np.linalg.LinAlgError:
        v2 = vc[:, :2]
    # Normalize to unit disk
    r = np.linalg.norm(v2, axis=1).max()
    if r > 0:
        v2 = v2 * (0.95 / r)

    parents = item["parents"]
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", lw=0.5)
    ax.add_patch(circle)
    # Edges
    for child in range(n):
        p = int(parents[child])
        if p < 0:
            continue
        ax.plot([v2[p, 0], v2[child, 0]], [v2[p, 1], v2[child, 1]],
                color="lightgray", lw=0.3, zorder=1)
    # Nodes colored by depth
    depths = np.asarray(item["depths"], dtype=np.float32)
    sc = ax.scatter(v2[:, 0], v2[:, 1], c=depths, cmap="viridis", s=8, zorder=2)
    plt.colorbar(sc, ax=ax, label="depth")
    # Highlight successes
    succ_mask = np.asarray([False] * n)
    # is_terminal etc. not loaded here — re-derive: only leaves if available
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Tree viz (tangent-PCA) — problem={item['problem']} n={n}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/head.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(config["training"]["output_dir"]) / "head.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    head = HyperbolicHead(
        in_dim=ckpt["in_dim"],
        hyp_dim=config["model"]["hyp_dim"],
        hidden_dims=config["model"]["head_hidden_dims"],
        manifold=config["model"]["manifold"],
    ).to(device).float()
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    out_dir = Path(config["eval"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "manifold": config["model"]["manifold"],
        "loss": config["training"]["loss"],
        "hyp_dim": config["model"]["hyp_dim"],
    }
    pairs_per_tree = int(config["eval"]["eval_pairs_per_tree"])
    for split in (config["data"]["val_split"], config["data"]["test_split"]):
        ds = TreeCacheDataset(config["data"]["tree_dir"], split)
        res = evaluate_split(head, ds, device, pairs_per_tree)
        save_scatter(res.pop("d_tree_flat"), res.pop("d_hyp_flat"),
                     out_dir / f"scatter_{split}.png")
        summary[split] = res
        print(f"[{split}] {res}", flush=True)

    # Visualizations on a few train trees
    vis_ds = TreeCacheDataset(config["data"]["tree_dir"], config["data"]["train_split"])
    for tidx in config["eval"].get("vis_problems", []):
        if tidx >= len(vis_ds):
            continue
        visualize_tree_2d(head, vis_ds, tidx, device,
                          out_dir / f"vis_tree_{tidx}.png")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
