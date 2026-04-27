"""Cache hidden states + tree topology for varied-target Game-24 problems.

For each (pool, target) record in the varied-target JSONL, enumerate the tree,
render every node's state text, forward through the frozen Qwen2.5-14B base,
save per-node hidden states and BFS distances to success.

Output layout (matches data/cd_trees_qwen14b/):
  data/24_varied_trees_qwen14b/{split}/problem_{i}.pt  — topology + distances
  data/24_varied_trees_qwen14b/{split}/hidden_{i}.npy  — float16 memmap [N, H]

Resume-on-existing: skip problems whose output files already exist.

Multi-GPU: pass --shard_rank and --shard_world. Each rank processes records
where i % world_size == rank.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tree_data_generic import (
    enumerate_tree_generic,
    render_tree_node,
    bfs_distances_to_success,
)


@torch.no_grad()
def encode_batch(model, tok, texts: list[str], device, max_len: int = 256):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(device)
    # last-token last-layer hidden
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]  # [B, T, H]
    # grab last non-pad token per row
    lens = enc["attention_mask"].sum(dim=1) - 1
    idx = lens.view(-1, 1, 1).expand(-1, 1, h.size(-1))
    last = h.gather(1, idx).squeeze(1)  # [B, H]
    return last.to(torch.float16).cpu().numpy()


def subsample_indices(n_total: int, is_success: np.ndarray,
                      dist: np.ndarray, cap: int, rng: random.Random
                      ) -> np.ndarray:
    """Pick a subset of node indices up to `cap`.

    Always keep ALL success terminals (dist=0). From the rest, stratify by
    tree distance so we see nodes at multiple BFS depths. Unreachable nodes
    (dist == INF) are dropped.
    """
    INF = 10 ** 8
    reachable = np.where(dist < INF)[0]
    if len(reachable) <= cap:
        return reachable

    success_idx = [i for i in reachable if is_success[i]]
    other_idx = [i for i in reachable if not is_success[i]]

    # Stratify other_idx by dist value
    buckets: dict[int, list[int]] = {}
    for i in other_idx:
        buckets.setdefault(int(dist[i]), []).append(i)

    remaining_cap = cap - len(success_idx)
    distinct_dists = sorted(buckets.keys())
    per_bucket = max(1, remaining_cap // max(1, len(distinct_dists)))

    chosen: list[int] = list(success_idx)
    for d in distinct_dists:
        bucket = buckets[d]
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_bucket])
        if len(chosen) >= cap:
            break

    # Top up if under cap (some buckets were small)
    if len(chosen) < cap:
        leftovers: list[int] = []
        for d in distinct_dists:
            leftovers.extend(buckets[d][per_bucket:])
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: cap - len(chosen)])
    return np.array(sorted(chosen), dtype=np.int32)


def process_one(rec: dict, out_dir: Path, model, tok, device,
                batch_size: int, max_len: int, node_cap: int,
                rng: random.Random) -> tuple[int, float]:
    idx = rec["_idx"]
    pt_path = out_dir / f"problem_{idx}.pt"
    h_path = out_dir / f"hidden_{idx}.npy"
    if pt_path.exists() and h_path.exists():
        return (0, 0.0)

    t0 = time.time()
    tree = enumerate_tree_generic(rec["pool"], rec["target"])
    n_total = len(tree.nodes)

    # Topology (always stored in full — cheap and useful for training-time
    # tree walks).
    parents = np.array([(n.parent if n.parent is not None else -1)
                        for n in tree.nodes], dtype=np.int32)
    depths = np.array([n.depth for n in tree.nodes], dtype=np.int16)
    is_terminal = np.array([n.is_terminal for n in tree.nodes], dtype=bool)
    is_success = np.array([n.is_success for n in tree.nodes], dtype=bool)
    dist = np.array(bfs_distances_to_success(tree), dtype=np.int32)

    if node_cap > 0 and n_total > node_cap:
        sampled_idx = subsample_indices(n_total, is_success, dist, node_cap,
                                        rng)
    else:
        sampled_idx = np.arange(n_total, dtype=np.int32)

    # Forward only sampled nodes
    sampled_texts = [render_tree_node(tree, tree.nodes[i]) for i in sampled_idx]
    H: list[np.ndarray] = []
    for s in range(0, len(sampled_texts), batch_size):
        batch = sampled_texts[s: s + batch_size]
        H.append(encode_batch(model, tok, batch, device, max_len=max_len))
    hidden = np.concatenate(H, axis=0)  # [k, H]  where k = len(sampled_idx)

    # v_values: drop-in alias for `dist_to_success`, using -1 for unreachable
    # so it matches the convention expected by src/train_head.py's
    # origin_ranking_loss (v_target < 0 → filtered out).
    INF = 10 ** 8
    v_values = np.where(dist >= INF, -1, dist).astype(np.int32)

    meta = {
        "pool": list(rec["pool"]),
        "target": int(rec["target"]),
        "n_steps": int(rec["n_steps"]),
        "source_problem": rec.get("source_problem", ""),
        "n": n_total,
        "parents": parents,
        "depths": depths,
        "is_terminal": is_terminal,
        "is_success": is_success,
        "dist_to_success": dist,
        "v_values": v_values,
        "sampled_idx": sampled_idx,    # length k; hidden[i] corresponds to
                                        # tree node sampled_idx[i]
    }
    torch.save(meta, pt_path)
    np.save(h_path, hidden)
    return (len(sampled_idx), time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--node_cap", type=int, default=200,
                    help="Max nodes per tree to cache. 0 = no cap. d3 trees "
                         "have ~4400 nodes, which makes full caching very "
                         "slow; 200 preserves all successes + a stratified "
                         "sample of other depths.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed + args.shard_rank)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rank {args.shard_rank}/{args.shard_world}] "
          f"loading {args.model} bf16 on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    records: list[dict] = []
    with open(args.jsonl) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r["_idx"] = i
            records.append(r)
    if args.limit > 0:
        records = records[: args.limit]

    my_records = [r for r in records
                  if r["_idx"] % args.shard_world == args.shard_rank]
    print(f"[rank {args.shard_rank}] {len(my_records)} records to process",
          flush=True)

    total_nodes = 0
    total_time = 0.0
    done = 0
    t0 = time.time()
    for r in my_records:
        n_nodes, t = process_one(r, out_dir, model, tok, device,
                                 args.batch_size, args.max_len,
                                 args.node_cap, rng)
        total_nodes += n_nodes
        total_time += t
        done += 1
        if done % 20 == 0 or done == len(my_records):
            wall = time.time() - t0
            rate = total_nodes / max(wall, 1e-6)
            print(f"[rank {args.shard_rank}] {done}/{len(my_records)}  "
                  f"nodes={total_nodes}  "
                  f"rate={rate:.0f} nodes/s  elapsed={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
