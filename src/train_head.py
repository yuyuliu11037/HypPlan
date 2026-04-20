"""Stage-1 trainer: fit the hyperbolic head so d_H(z_i, z_j) ≈ d_tree(i, j).

LLM is NOT loaded — we consume pre-cached last-token hidden states from
data/trees/{split}/. Each "sample" is one tree; we sample pairs_per_tree
pairs per tree and compute distortion or ranking loss.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.head import HyperbolicHead
from src.tree_data import pair_distances_lca


class TreeCacheDataset(Dataset):
    """One sample = one cached tree (metadata + hidden-state memmap)."""

    def __init__(self, tree_dir: str, split: str):
        d = Path(tree_dir) / split
        files = glob(str(d / "problem_*.pt"))
        assert len(files) > 0, f"no trees in {d}"
        # Numeric sort so problem_10 comes after problem_9 (glob is lex order)
        self.indices = sorted(int(Path(p).stem.split("_")[1]) for p in files)
        self.split_dir = d

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        meta = torch.load(self.split_dir / f"problem_{idx}.pt", weights_only=False)
        hidden = np.load(self.split_dir / f"hidden_{idx}.npy")
        # Countdown caches store v_values (continuous |final - target|) and
        # pool/target; Game-24 caches don't have these. Pass through if present.
        out = {
            "idx": idx,
            "problem": meta.get("problem", str(meta.get("pool", ""))),
            "n": int(meta["n"]),
            "parents": meta["parents"],
            "depths": meta["depths"],
            "is_terminal": meta["is_terminal"],
            "is_success": meta["is_success"],
            "hidden": hidden,  # float16 (n, H)
        }
        if "v_values" in meta:
            out["v_values"] = meta["v_values"]
        return out


def _collate_passthrough(batch):
    return batch  # list of dicts


def sample_pairs(n: int, k: int, rng: random.Random) -> np.ndarray:
    """K random pairs (i < j) sampled with replacement from upper triangle.

    With replacement is fine — for large N, repeat prob is tiny; for small N,
    the tree is small anyway and we'd want coverage.
    """
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    k = min(k, n * (n - 1) // 2) if n < 64 else k  # cap for tiny trees
    i = np.random.randint(0, n, size=k)
    j = np.random.randint(0, n, size=k)
    mask = i == j
    # Resolve self-pairs by bumping j
    j[mask] = (j[mask] + 1) % n
    # Swap to ensure i < j
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return np.stack([lo, hi], axis=1)


def distortion_loss(
    head: HyperbolicHead, z: torch.Tensor, pairs: np.ndarray,
    d_tree: np.ndarray,
) -> torch.Tensor:
    i = torch.as_tensor(pairs[:, 0], device=z.device, dtype=torch.long)
    j = torch.as_tensor(pairs[:, 1], device=z.device, dtype=torch.long)
    d_hyp = head.distance(z[i], z[j])
    d_tree_t = torch.as_tensor(d_tree, device=z.device, dtype=torch.float32)
    return ((d_hyp - d_tree_t) ** 2).mean()


def distance_to_nearest_solution(parents: np.ndarray,
                                  is_success: np.ndarray) -> np.ndarray:
    """BFS from all success leaves outward over the undirected tree.

    Returns int array v where v[i] = min edge-count to any success leaf.
    v[i] = -1 means "unreachable" (tree has no success leaves, or
    disconnected components — shouldn't happen on a connected tree).
    """
    from collections import deque
    n = parents.shape[0]
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        p = int(parents[i])
        if p >= 0:
            children[p].append(i)

    dist = np.full(n, -1, dtype=np.int64)
    q = deque()
    for i in range(n):
        if bool(is_success[i]):
            dist[i] = 0
            q.append(i)
    while q:
        u = q.popleft()
        p = int(parents[u])
        neigh = children[u] + ([p] if p >= 0 else [])
        for v in neigh:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def origin_ranking_loss(
    head: HyperbolicHead, z: torch.Tensor,
    v_target: np.ndarray, n_pairs: int, margin: float, rng: random.Random,
) -> torch.Tensor:
    """Margin ranking loss with the origin as anchor.

    For each sampled pair (i, j) with v[i] < v[j] (i is closer to a
    solution), enforce
        d_H(z_i, 0) + margin <= d_H(z_j, 0)
    via hinge loss max(0, d_i - d_j + margin). Pairs with equal targets
    contribute zero; nodes with v=-1 (unreachable) are filtered out.
    """
    valid = np.where(v_target >= 0)[0]
    if len(valid) < 2:
        return torch.zeros((), device=z.device, requires_grad=True)

    i = valid[np.random.randint(0, len(valid), size=n_pairs)]
    j = valid[np.random.randint(0, len(valid), size=n_pairs)]
    diff = v_target[i] != v_target[j]
    if not diff.any():
        return torch.zeros((), device=z.device, requires_grad=True)
    i, j = i[diff], j[diff]
    # Orient so v[i] < v[j]
    flip = v_target[i] > v_target[j]
    if flip.any():
        i_new = np.where(flip, j, i)
        j_new = np.where(flip, i, j)
        i, j = i_new, j_new

    i_t = torch.as_tensor(i, device=z.device, dtype=torch.long)
    j_t = torch.as_tensor(j, device=z.device, dtype=torch.long)
    d_i = head.origin_distance(z[i_t])
    d_j = head.origin_distance(z[j_t])
    return torch.relu(d_i - d_j + margin).mean()


def origin_ranking_rank_loss(
    head: HyperbolicHead, z: torch.Tensor,
    v_raw: np.ndarray, n_pairs: int, margin: float, rng: random.Random,
) -> torch.Tensor:
    """Rank-based origin_ranking loss for wide-dynamic-range v (e.g. Countdown
    where v = |final − target| can span 0 to millions).

    Scale-invariant: sort v (reachable nodes only), sample pair indices,
    enforce `d_H(z_i, 0) + margin <= d_H(z_j, 0)` when rank[i] < rank[j].
    Strict rank ordering — ties (same v) are filtered out to keep the signal
    clean.
    """
    valid_idx = np.where(v_raw >= 0)[0]
    if len(valid_idx) < 2:
        return torch.zeros((), device=z.device, requires_grad=True)

    # Stable ascending sort by v; return the indices into the original array.
    v_valid = v_raw[valid_idx]
    order = valid_idx[np.argsort(v_valid, kind="stable")]
    v_sorted = v_raw[order]  # ascending

    # Sample rank pairs (a < b in sorted order)
    m = len(order)
    a = np.random.randint(0, m, size=n_pairs)
    b = np.random.randint(0, m, size=n_pairs)
    mask = a != b
    if not mask.any():
        return torch.zeros((), device=z.device, requires_grad=True)
    a, b = a[mask], b[mask]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    # Drop ties in raw v (same-rank pair provides no signal)
    neq = v_sorted[lo] != v_sorted[hi]
    if not neq.any():
        return torch.zeros((), device=z.device, requires_grad=True)
    lo, hi = lo[neq], hi[neq]

    i_ids = order[lo]  # closer to target (smaller v, smaller |z|)
    j_ids = order[hi]  # farther (larger v, larger |z|)

    i_t = torch.as_tensor(i_ids, device=z.device, dtype=torch.long)
    j_t = torch.as_tensor(j_ids, device=z.device, dtype=torch.long)
    d_i = head.origin_distance(z[i_t])
    d_j = head.origin_distance(z[j_t])
    return torch.relu(d_i - d_j + margin).mean()


def ranking_loss(
    head: HyperbolicHead, z: torch.Tensor,
    parents: np.ndarray, n: int, num_negatives: int, rng: random.Random,
) -> torch.Tensor:
    """Nickel-Kiela: for each (parent, child) edge, child must be closer than
    K random non-descendant negatives.

    Approximation for efficiency: use "not-the-child, not-the-parent" as a
    cheap negative pool (a small fraction are descendants; acceptable noise).
    """
    child_ids_np = np.where(parents != -1)[0]
    if len(child_ids_np) == 0:
        return torch.zeros((), device=z.device, requires_grad=True)
    parent_ids_np = parents[child_ids_np]
    edges = np.stack([parent_ids_np, child_ids_np], axis=1)

    E = len(edges)
    # Sample K random negatives per edge from all nodes (bias excluded: the
    # child itself). We accept a tiny chance of the negative being a descendant.
    neg = np.random.randint(0, n, size=(E, num_negatives))

    parent_ids = torch.as_tensor(edges[:, 0], device=z.device, dtype=torch.long)
    child_ids = torch.as_tensor(edges[:, 1], device=z.device, dtype=torch.long)
    neg_ids = torch.as_tensor(neg, device=z.device, dtype=torch.long)  # (E, K)

    # Distances
    d_pos = head.distance(z[parent_ids], z[child_ids])               # (E,)
    z_parents_rep = z[parent_ids].unsqueeze(1).expand(-1, num_negatives, -1)
    d_neg = head.distance(z_parents_rep, z[neg_ids])                  # (E, K)

    # Softmax over {pos, neg} with negative-distance as score
    neg_scores = -d_neg
    pos_scores = -d_pos.unsqueeze(1)                                  # (E, 1)
    all_scores = torch.cat([pos_scores, neg_scores], dim=1)           # (E, K+1)
    log_probs = torch.log_softmax(all_scores, dim=1)
    return -log_probs[:, 0].mean()


def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return distributed, rank, world_size, local_rank, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/head.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, rank, world_size, local_rank, device = setup_distributed()
    rng = random.Random(42 + rank)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)

    # Dataset
    train_ds = TreeCacheDataset(config["data"]["tree_dir"], config["data"]["train_split"])
    val_ds = TreeCacheDataset(config["data"]["tree_dir"], config["data"]["val_split"])

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=_collate_passthrough,
        num_workers=2, pin_memory=False,
    )

    # Infer in_dim from first hidden
    first = train_ds[0]
    in_dim = first["hidden"].shape[1]

    head = HyperbolicHead(
        in_dim=in_dim,
        hyp_dim=config["model"]["hyp_dim"],
        hidden_dims=config["model"]["head_hidden_dims"],
        manifold=config["model"]["manifold"],
    ).to(device).float()

    if distributed:
        head_ddp = torch.nn.parallel.DistributedDataParallel(
            head, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
        head_train = head_ddp
        head_ref = head_ddp.module
    else:
        head_train = head
        head_ref = head

    # Optim
    loss_name = config["training"]["loss"]
    assert loss_name in ("distortion", "ranking", "origin_ranking",
                         "origin_ranking_rank")
    margin = float(config["training"].get("origin_ranking_margin", 1.0))
    lr = float(config["training"]["lr"])
    wd = float(config["training"].get("weight_decay", 1e-4))
    epochs = int(config["training"]["epochs"])
    trees_per_batch = int(config["training"]["trees_per_batch"])
    pairs_per_tree = int(config["data"]["pairs_per_tree"])
    num_negatives = int(config["training"].get("ranking_negatives", 10))

    steps_per_epoch = max(len(train_loader) // trees_per_batch, 1)
    total_steps = steps_per_epoch * epochs
    warmup = int(total_steps * float(config["training"].get("warmup_ratio", 0.05)))

    optimizer = AdamW(head_train.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    out_dir = Path(config["training"]["output_dir"])
    log_dir = Path(config["training"]["log_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    log_file = log_dir / f"head_{config['model']['manifold']}_{loss_name}_train.jsonl"
    if rank == 0:
        print(f"train_trees={len(train_ds)} val_trees={len(val_ds)} "
              f"total_steps={total_steps} loss={loss_name} "
              f"manifold={config['model']['manifold']} hyp_dim={config['model']['hyp_dim']}",
              flush=True)

    global_step = 0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        accum_loss_sum = 0.0
        accum_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            item = batch[0]
            n = item["n"]
            if n < 2:
                continue
            hidden = torch.as_tensor(item["hidden"], device=device, dtype=torch.float32)
            z = head_train(hidden)

            if loss_name == "distortion":
                pairs = sample_pairs(n, pairs_per_tree, rng)
                d_tree = pair_distances_lca(
                    item["parents"], item["depths"],
                    pairs[:, 0], pairs[:, 1],
                ).astype(np.float32)
                loss = distortion_loss(head_ref, z, pairs, d_tree)
            elif loss_name == "ranking":
                loss = ranking_loss(
                    head_ref, z, item["parents"], n, num_negatives, rng,
                )
            elif loss_name == "origin_ranking":
                v_target = distance_to_nearest_solution(
                    item["parents"], item["is_success"],
                )
                loss = origin_ranking_loss(
                    head_ref, z, v_target, pairs_per_tree, margin, rng,
                )
            else:  # origin_ranking_rank (Countdown: continuous v, rank-based)
                v_raw = item.get("v_values")
                assert v_raw is not None, (
                    "origin_ranking_rank requires v_values in tree meta "
                    "(Countdown pipeline)")
                loss = origin_ranking_rank_loss(
                    head_ref, z, np.asarray(v_raw), pairs_per_tree, margin, rng,
                )

            (loss / trees_per_batch).backward()
            accum_loss_sum += float(loss.item())
            accum_count += 1

            if (batch_idx + 1) % trees_per_batch == 0:
                torch.nn.utils.clip_grad_norm_(head_train.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and (global_step % 25 == 0 or global_step == 1):
                    avg = accum_loss_sum / max(accum_count, 1)
                    z_norm = float(z.detach().float().norm(dim=-1).mean())
                    print(
                        f"epoch {epoch} step {global_step}/{total_steps} "
                        f"loss={avg:.4f} z_norm={z_norm:.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"n={n}",
                        flush=True,
                    )
                    with open(log_file, "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "epoch": epoch,
                            "loss": round(avg, 4), "z_norm": round(z_norm, 4),
                            "lr": scheduler.get_last_lr()[0], "n": n,
                        }) + "\n")

                accum_loss_sum = 0.0
                accum_count = 0

        if rank == 0:
            elapsed = time.time() - epoch_start
            print(f"== epoch {epoch} done in {elapsed:.1f}s ==", flush=True)

    if rank == 0:
        ckpt = {
            "state_dict": head_ref.state_dict(),
            "config": config,
            "in_dim": in_dim,
        }
        torch.save(ckpt, out_dir / "head.pt")
        print(f"Saved head to {out_dir / 'head.pt'}", flush=True)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
