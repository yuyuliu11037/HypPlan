"""Build graph-coloring tree caches for Stage-1 head training.

Same pattern as ProntoQA / Blocksworld. Reads
data/graphcolor_problems.json (split assignments) and writes:
  data/graphcolor_trees_qwen14b/{train,val,test}/problem_<idx>.{pt,_npy}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_graphcolor import (
    Problem, enumerate_tree, render_state,
)


def build_one(prob_dict, model, tokenizer, device) -> tuple[dict, np.ndarray]:
    p = Problem(n=prob_dict["n"], edges=tuple(map(tuple, prob_dict["edges"])),
                  one_solution=tuple(prob_dict["gold"]))
    tree = enumerate_tree(p, max_nodes=2000)
    n = len(tree.nodes)
    parents = np.full(n, -1, dtype=np.int32)
    depths = np.zeros(n, dtype=np.int16)
    is_terminal = np.zeros(n, dtype=bool)
    is_success = np.zeros(n, dtype=bool)
    v_values = np.full(n, -1, dtype=np.int32)
    for nd in tree.nodes:
        if nd.parent is not None:
            parents[nd.node_id] = nd.parent
        depths[nd.node_id] = nd.depth
        is_terminal[nd.node_id] = (len(nd.children) == 0)
        is_success[nd.node_id] = nd.is_complete
        v_values[nd.node_id] = nd.v_value

    state_texts = [render_state(p, nd.state) for nd in tree.nodes]
    hiddens = np.zeros((n, model.config.hidden_size), dtype=np.float16)
    BATCH = 16
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
        "v_values": v_values, "problem": f"gc_n{p.n}_e{len(p.edges)}",
    }
    return meta, hiddens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/graphcolor_trees_qwen14b")
    ap.add_argument("--problems", default="data/graphcolor_problems.json")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    out = Path(args.out_dir)
    for s in args.splits.split(","):
        (out / s).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[r{args.shard_rank}] Loading {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                                trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.base_model,
                                       torch_dtype=torch.bfloat16,
                                       trust_remote_code=True).to(device).eval()

    all_problems = json.load(open(args.problems))
    by_split: dict = {"train": [], "val": [], "test": []}
    for p in all_problems:
        by_split[p["split"]].append(p)

    for split_name in args.splits.split(","):
        recs = by_split[split_name]
        shard = [(i, r) for (i, r) in enumerate(recs)
                 if i % args.shard_world == args.shard_rank]
        print(f"[r{args.shard_rank}] {split_name}: {len(shard)} of {len(recs)}",
              flush=True)
        for ti, (orig_idx, rec) in enumerate(shard):
            out_pt = out / split_name / f"problem_{orig_idx}.pt"
            out_npy = out / split_name / f"hidden_{orig_idx}.npy"
            if out_pt.exists() and out_npy.exists():
                continue
            t0 = time.time()
            meta, hiddens = build_one(rec, model, tokenizer, device)
            torch.save(meta, out_pt); np.save(out_npy, hiddens)
            if (ti + 1) % 10 == 0:
                print(f"  [r{args.shard_rank}/{split_name}] "
                       f"{ti+1}/{len(shard)} problem_{orig_idx}: "
                       f"{meta['n']} nodes in {time.time()-t0:.1f}s",
                       flush=True)


if __name__ == "__main__":
    main()
