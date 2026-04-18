"""Value probe: does the trained geometric head encode tree-value information?

For each node in the cached training trees, compute a scalar "value" equal to
  (# descendant leaves with is_success) / (# descendant leaves)
which is 0 for sub-trees that never reach 24 and 1 for sub-trees all of whose
leaves reach 24. Run the *trained* head on the cached hidden states to get z,
then:
  (a) linear probe z -> value, report R^2 (train / test split of nodes).
  (b) correlation between hyperbolic norm |z| (= d_H(0, z)) and value.

No training of the head happens here; the head is frozen.

Usage:
  python -m src.value_probe --head_tag poincare_distortion --split train
  python -m src.value_probe --head_tag lorentz_distortion  --split train
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from src.head import HyperbolicHead, poincare_distance
from src.hyperbolic import lorentz_distance
from src.train_head import TreeCacheDataset


def compute_node_values(parents: np.ndarray, is_terminal: np.ndarray,
                        is_success: np.ndarray) -> np.ndarray:
    """For each node, value = (# descendant leaves with is_success) / (# descendant leaves).

    A node counts as its own descendant. Leaves have value 0 or 1.
    Internal nodes with no leaf descendants get NaN (shouldn't happen for our trees).
    """
    n = parents.shape[0]
    # Children lists from parent pointers
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        p = int(parents[i])
        if p >= 0:
            children[p].append(i)

    # Iterative post-order: compute (leaf_total[i], leaf_success[i]).
    leaf_total = np.zeros(n, dtype=np.int64)
    leaf_success = np.zeros(n, dtype=np.int64)

    # Topological order by depth (parent before child -> reverse for post-order)
    # Easier: process nodes in reverse BFS order. parents[0] = -1, and children
    # always have id > parent id in our builder, so reverse-index order works.
    for i in reversed(range(n)):
        if is_terminal[i]:
            leaf_total[i] = 1
            leaf_success[i] = 1 if is_success[i] else 0
        else:
            ch = children[i]
            if ch:
                leaf_total[i] = leaf_total[ch].sum()
                leaf_success[i] = leaf_success[ch].sum()
            # else: internal w/ no children -- leave as 0

    with np.errstate(invalid="ignore", divide="ignore"):
        val = leaf_success.astype(np.float64) / np.maximum(leaf_total, 1)
    val[leaf_total == 0] = np.nan
    return val, leaf_total, leaf_success


def origin_distance(z: torch.Tensor, manifold: str) -> torch.Tensor:
    """Hyperbolic distance from origin for a batch of manifold points."""
    if manifold == "poincare":
        zero = torch.zeros_like(z)
        return poincare_distance(z, zero)
    elif manifold == "lorentz":
        zero = torch.zeros_like(z)
        zero[..., 0] = 1.0  # origin on the hyperboloid is (1, 0, ..., 0)
        return lorentz_distance(z, zero)
    else:
        raise ValueError(manifold)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    denom = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y, nan_policy="omit").correlation)
    except Exception:
        def rankdata(a):
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(a) + 1)
            return ranks
        return pearson(rankdata(x), rankdata(y))


def linear_probe_r2(Z: np.ndarray, y: np.ndarray, seed: int = 0,
                    test_frac: float = 0.2, ridge: float = 1e-3) -> dict:
    """Closed-form ridge probe. Returns train/test R^2."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    Ztr, ytr = Z[tr], y[tr]
    Zte, yte = Z[te], y[te]

    # Add bias column
    def aug(X): return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    Xtr = aug(Ztr); Xte = aug(Zte)
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + ridge * np.eye(d)
    b = Xtr.T @ ytr
    w = np.linalg.solve(A, b)
    ypred_tr = Xtr @ w
    ypred_te = Xte @ w

    def r2(y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "r2_train": r2(ytr, ypred_tr),
        "r2_test": r2(yte, ypred_te),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_tag", required=True,
                    help="e.g. poincare_distortion or lorentz_distortion")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--tree_dir", default="data/trees")
    ap.add_argument("--max_trees", type=int, default=None)
    ap.add_argument("--max_nodes_per_tree", type=int, default=None,
                    help="Optional cap; if set, random-subsample nodes per tree.")
    ap.add_argument("--output",
                    default=None,
                    help="Output JSON path. Defaults to results/value_probe/{tag}_{split}.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path("checkpoints") / f"head_{args.head_tag}"
    cfg = yaml.safe_load(open(ckpt_dir / "config.yaml"))
    manifold = cfg["model"]["manifold"]
    hyp_dim = cfg["model"]["hyp_dim"]
    hidden_dims = cfg["model"]["head_hidden_dims"]

    ds = TreeCacheDataset(args.tree_dir, args.split)
    print(f"split={args.split} n_trees={len(ds)} manifold={manifold} hyp_dim={hyp_dim}")

    # Peek hidden dim from first tree
    h0 = ds[0]["hidden"]
    in_dim = int(h0.shape[-1])

    head = HyperbolicHead(in_dim=in_dim, hyp_dim=hyp_dim,
                          hidden_dims=hidden_dims, manifold=manifold)
    sd = torch.load(ckpt_dir / "head.pt", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    head.load_state_dict(sd)
    head.eval().to(args.device)

    zs: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    values: list[np.ndarray] = []
    leaf_totals: list[np.ndarray] = []

    n_trees = len(ds) if args.max_trees is None else min(args.max_trees, len(ds))
    rng = np.random.default_rng(0)
    for ti in range(n_trees):
        item = ds[ti]
        n = item["n"]
        if n < 2:
            continue
        # TreeCacheDataset doesn't surface is_success; reload meta.
        meta = torch.load(ds.split_dir / f"problem_{ds.indices[ti]}.pt",
                          weights_only=False)
        val, lt, ls = compute_node_values(item["parents"], item["is_terminal"],
                                          meta["is_success"])
        hidden = torch.as_tensor(item["hidden"], device=args.device, dtype=torch.float32)

        # Forward in chunks to avoid spiking mem on big trees
        CHUNK = 4096
        z_parts = []
        norm_parts = []
        for s in range(0, n, CHUNK):
            z = head(hidden[s:s + CHUNK])
            d0 = origin_distance(z, manifold)
            z_parts.append(z.float().cpu().numpy())
            norm_parts.append(d0.float().cpu().numpy())
        z_all = np.concatenate(z_parts, axis=0)
        norm_all = np.concatenate(norm_parts, axis=0)

        # Optional subsample
        if args.max_nodes_per_tree is not None and n > args.max_nodes_per_tree:
            pick = rng.choice(n, size=args.max_nodes_per_tree, replace=False)
            z_all = z_all[pick]
            norm_all = norm_all[pick]
            val = val[pick]
            lt = lt[pick]

        zs.append(z_all)
        norms.append(norm_all)
        values.append(val)
        leaf_totals.append(lt)

        if (ti + 1) % 50 == 0:
            print(f"  processed {ti + 1}/{n_trees} trees")

    Z = np.concatenate(zs, axis=0)
    N = np.concatenate(norms, axis=0)
    Y = np.concatenate(values, axis=0)
    LT = np.concatenate(leaf_totals, axis=0)

    mask = np.isfinite(Y)
    Z = Z[mask]; N = N[mask]; Y = Y[mask]; LT = LT[mask]
    print(f"total nodes={len(Y)} mean(value)={Y.mean():.4f} "
          f"frac(value>0)={(Y>0).mean():.4f} frac(value==1)={(Y==1).mean():.4f}")

    probe = linear_probe_r2(Z, Y.astype(np.float64))
    r_norm = pearson(N.astype(np.float64), Y.astype(np.float64))
    rs_norm = spearman(N.astype(np.float64), Y.astype(np.float64))

    # Sanity baselines: predict from log(leaf_total) alone, predict mean
    def r2_mean(y): return 0.0  # by construction
    # How much does leaf_total explain of value? (near 0 ideally — unrelated)
    r_lt = pearson(np.log1p(LT).astype(np.float64), Y.astype(np.float64))

    # Also probe on non-leaf nodes only (where value is continuous, not binary)
    # Leaves have value in {0, 1}; interior nodes are averages -> more informative test.
    nonleaf = LT > 1
    if nonleaf.sum() > 100:
        probe_nonleaf = linear_probe_r2(Z[nonleaf], Y[nonleaf].astype(np.float64))
        r_norm_nonleaf = pearson(N[nonleaf].astype(np.float64), Y[nonleaf].astype(np.float64))
        rs_norm_nonleaf = spearman(N[nonleaf].astype(np.float64), Y[nonleaf].astype(np.float64))
    else:
        probe_nonleaf = None
        r_norm_nonleaf = None
        rs_norm_nonleaf = None

    result = {
        "head_tag": args.head_tag,
        "split": args.split,
        "manifold": manifold,
        "hyp_dim": hyp_dim,
        "n_nodes_total": int(len(Y)),
        "mean_value": float(Y.mean()),
        "frac_value_gt0": float((Y > 0).mean()),
        "frac_value_eq1": float((Y == 1).mean()),
        "linear_probe_all": probe,
        "pearson_norm_value": r_norm,
        "spearman_norm_value": rs_norm,
        "pearson_log_leaf_total_value": r_lt,
        "linear_probe_nonleaf": probe_nonleaf,
        "pearson_norm_value_nonleaf": r_norm_nonleaf,
        "spearman_norm_value_nonleaf": rs_norm_nonleaf,
        "n_nodes_nonleaf": int(nonleaf.sum()),
    }
    print(json.dumps(result, indent=2))

    out = args.output or f"results/value_probe/{args.head_tag}_{args.split}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
