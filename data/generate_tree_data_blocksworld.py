"""Build Blocksworld tree caches for Stage-1 head training.

Same pattern as `data/generate_tree_data_pronto.py` but using PlanBench's
blocksworld split and `src/oracle_blocksworld.py`.

Splits: deterministic shuffle of the 500 oneshot blocksworld records, slice
250 train / 50 val / 200 test (the test 200 stay disjoint from anything used
in head training; same indices as the eval test set).
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
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_blocksworld import enumerate_tree, parse_problem, render_state


def split_records(seed: int = 1234):
    bw = load_dataset("tasksource/planbench", "task_1_plan_generation",
                       split="train")
    bw_recs = [ex for ex in bw if ex["domain"] == "blocksworld"
               and ex["prompt_type"] == "oneshot"]
    rng = random.Random(seed)
    rng.shuffle(bw_recs)
    train = bw_recs[200:450]   # records 200-449 (test is 0-199 by same seed)
    val = bw_recs[450:500]
    test = bw_recs[:200]
    return train, val, test


def build_one(rec, model, tokenizer, device) -> tuple[dict, np.ndarray]:
    p = parse_problem(rec["query"])
    tree = enumerate_tree(p, max_nodes=2000)

    n = len(tree.nodes)
    parents = np.full(n, -1, dtype=np.int32)
    depths = np.zeros(n, dtype=np.int16)
    is_terminal = np.zeros(n, dtype=bool)
    is_success = np.zeros(n, dtype=bool)
    v_values = np.full(n, -1, dtype=np.int32)
    for node in tree.nodes:
        if node.parent is not None:
            parents[node.node_id] = node.parent
        depths[node.node_id] = node.depth
        is_terminal[node.node_id] = (len(node.children) == 0)
        is_success[node.node_id] = node.is_goal
        v_values[node.node_id] = node.v_value

    state_texts = [render_state(p, node.state) for node in tree.nodes]
    hiddens = np.zeros((n, model.config.hidden_size), dtype=np.float16)
    BATCH = 8
    with torch.no_grad():
        for s in range(0, n, BATCH):
            batch = state_texts[s : s + BATCH]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(device)
            out = model(**enc, output_hidden_states=False)
            last_h = out.last_hidden_state
            attn = enc["attention_mask"].sum(-1) - 1
            for bi in range(last_h.size(0)):
                hiddens[s + bi] = last_h[bi, int(attn[bi]), :].float().cpu().numpy()

    meta = {
        "n": n, "parents": parents, "depths": depths,
        "is_terminal": is_terminal, "is_success": is_success,
        "v_values": v_values, "problem": str(rec["instance_id"]),
    }
    return meta, hiddens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/blocksworld_trees_qwen14b")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.splits.split(","):
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[r{args.shard_rank}] Loading {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                                trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.base_model,
                                       torch_dtype=torch.bfloat16,
                                       trust_remote_code=True).to(device).eval()

    train, val, test = split_records()
    splits = {"train": train, "val": val, "test": test}

    for split_name in args.splits.split(","):
        recs = splits[split_name]
        if args.limit > 0:
            recs = recs[: args.limit]
        shard = [(i, r) for (i, r) in enumerate(recs)
                 if i % args.shard_world == args.shard_rank]
        print(f"[r{args.shard_rank}] {split_name}: {len(shard)} of "
              f"{len(recs)}", flush=True)
        for ti, (orig_idx, rec) in enumerate(shard):
            out_pt = out_dir / split_name / f"problem_{orig_idx}.pt"
            out_npy = out_dir / split_name / f"hidden_{orig_idx}.npy"
            if out_pt.exists() and out_npy.exists():
                continue
            t0 = time.time()
            try:
                meta, hiddens = build_one(rec, model, tokenizer, device)
            except Exception as e:
                print(f"  [r{args.shard_rank}] FAILED problem_{orig_idx}: {e}",
                       flush=True)
                continue
            torch.save(meta, out_pt)
            np.save(out_npy, hiddens)
            if (ti + 1) % 5 == 0:
                print(f"  [r{args.shard_rank}/{split_name}] {ti+1}/{len(shard)} "
                      f"problem_{orig_idx}: {meta['n']} nodes in "
                      f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
