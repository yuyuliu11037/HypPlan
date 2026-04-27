"""Build ProntoQA tree caches for Stage-1 head training.

For each problem in the train/val/test split, enumerate the forward-chaining
state tree, render each state as text, forward through the frozen base model,
and save (metadata, hidden_states) to disk.

Output layout:
  data/pronto_trees_qwen14b/
    {train,val,test}/
      problem_0.pt       # metadata
      hidden_0.npy       # float16 (n, H) hidden states
      ...

Metadata schema matches `src/train_head.py:TreeCacheDataset` expectations:
  {n, parents, depths, is_terminal, is_success, v_values, problem}

Splits: deterministic shuffle of the 500 ProntoQA validation records, then
slice 250 train / 50 val / 200 test. (We keep a 200-record test that stays
disjoint from anything used in head training.)
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
from src.oracle_pronto import enumerate_tree, parse_problem, render_state


def split_records(seed: int = 1234):
    """Return (train, val, test) lists of HF records, deterministic split."""
    ds = load_dataset("renma/ProntoQA", split="validation")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    train = [ds[i] for i in idx[:250]]
    val = [ds[i] for i in idx[250:300]]
    test = [ds[i] for i in idx[300:500]]
    return train, val, test


def build_one(rec, model, tokenizer, device) -> tuple[dict, np.ndarray]:
    p = parse_problem(rec["raw_logic_programs"])
    tree = enumerate_tree(p)

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
        is_success[node.node_id] = node.is_decidable
        v_values[node.node_id] = node.v_value

    # Forward each state's text through the base model.
    state_texts = [render_state(p, node.state) for node in tree.nodes]
    hiddens = np.zeros((n, model.config.hidden_size), dtype=np.float16)
    BATCH = 8
    with torch.no_grad():
        for s in range(0, n, BATCH):
            batch = state_texts[s : s + BATCH]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(device)
            out = model(**enc, output_hidden_states=False)
            # Mean-pool? Or last-token? We use last-non-pad token to match the
            # G24 head which uses last-token last-layer.
            last_h = out.last_hidden_state  # (B, T, H)
            attn = enc["attention_mask"].sum(-1) - 1   # (B,) idx of last real token
            for bi in range(last_h.size(0)):
                hiddens[s + bi] = last_h[bi, int(attn[bi]), :].float().cpu().numpy()

    meta = {
        "n": n, "parents": parents, "depths": depths,
        "is_terminal": is_terminal, "is_success": is_success,
        "v_values": v_values,
        "problem": rec["id"],
    }
    return meta, hiddens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/pronto_trees_qwen14b")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1,
                     help="Cap records per split (for smoke testing).")
    ap.add_argument("--splits", default="train,val,test")
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
        # Shard: each rank takes every `shard_world`-th record
        shard = [(i, r) for (i, r) in enumerate(recs)
                 if i % args.shard_world == args.shard_rank]
        print(f"[r{args.shard_rank}] {split_name}: {len(shard)} of "
              f"{len(recs)} records", flush=True)
        for ti, (orig_idx, rec) in enumerate(shard):
            out_pt = out_dir / split_name / f"problem_{orig_idx}.pt"
            out_npy = out_dir / split_name / f"hidden_{orig_idx}.npy"
            if out_pt.exists() and out_npy.exists():
                continue
            t0 = time.time()
            meta, hiddens = build_one(rec, model, tokenizer, device)
            torch.save(meta, out_pt)
            np.save(out_npy, hiddens)
            if (ti + 1) % 5 == 0:
                print(f"  [r{args.shard_rank}/{split_name}] {ti+1}/{len(shard)} "
                      f"problem_{orig_idx}: {meta['n']} nodes in "
                      f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
