"""Offline tree data generator for HypPlan v2.

For each solvable Game-of-24 problem in the existing train/val/test jsonl
splits, enumerate the full state tree (including dead ends), compute the
tree-distance matrix, and cache the frozen SFT LLM's last-token hidden state
for every node.

Outputs per split under data/trees/{split}/:
  problem_{idx}.pt    -- torch pickle of {"problem", "n", "parents",
                          "depths", "is_terminal", "is_success"}
  hidden_{idx}.npy    -- float16 memmap of shape (n, hidden_dim)

Distance matrix is NOT stored; distances are recomputed in-loop via
pair_distances_lca() from src/tree_data.py. Texts are NOT stored; re-render
from the tree (fast) when needed.

Resumes automatically: skips problems whose both output files already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tree_data import enumerate_tree, render_state


def load_problems(jsonl_path: str) -> list[str]:
    seen: set = set()
    problems: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                problems.append(p)
    return problems


@torch.no_grad()
def encode_texts(texts: list[str], tokenizer, model, device, batch_size: int) -> np.ndarray:
    """Return (len(texts), hidden_dim) float16 array of last-token last-layer states."""
    hidden_dim = model.config.hidden_size
    out = np.empty((len(texts), hidden_dim), dtype=np.float16)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt", add_special_tokens=True,
        ).to(device)
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, L, H)
        # last non-pad token index per row
        last_idx = enc["attention_mask"].sum(dim=1) - 1  # (B,)
        B = last_hidden.size(0)
        row = last_hidden[torch.arange(B, device=device), last_idx]  # (B, H)
        out[i : i + B] = row.float().cpu().numpy().astype(np.float16)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft_24_tot_merged")
    parser.add_argument("--out_dir", default="data/trees")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--split_files", nargs="+",
                        default=["data/24_train.jsonl",
                                 "data/24_val.jsonl",
                                 "data/24_test.jsonl"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=-1,
                        help="If >0, cap problems per split (for smoke tests)")
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--shard_world", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    assert len(args.splits) == len(args.split_files)

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # with sum(mask)-1 indexing below
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map={"": device},
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"hidden_dim = {hidden_dim}", flush=True)

    for split, split_file in zip(args.splits, args.split_files):
        problems = load_problems(split_file)
        if args.limit > 0:
            problems = problems[: args.limit]
        out_split = Path(args.out_dir) / split
        out_split.mkdir(parents=True, exist_ok=True)

        print(f"\n[{split}] {len(problems)} problems "
              f"(shard {args.shard_rank}/{args.shard_world})", flush=True)
        total_nodes = 0
        start = time.time()
        for idx, problem in enumerate(problems):
            # Shard: each rank handles indices where idx % world == rank
            if idx % args.shard_world != args.shard_rank:
                continue
            meta_path = out_split / f"problem_{idx}.pt"
            hidden_path = out_split / f"hidden_{idx}.npy"
            if meta_path.exists() and hidden_path.exists():
                total_nodes += torch.load(meta_path, weights_only=False)["n"]
                continue

            tree = enumerate_tree(problem)
            texts = [render_state(tree, node) for node in tree.nodes]
            hidden = encode_texts(texts, tokenizer, model, device, args.batch_size)

            parents = np.array(
                [n.parent if n.parent is not None else -1 for n in tree.nodes],
                dtype=np.int32,
            )
            depths = np.array([n.depth for n in tree.nodes], dtype=np.int16)
            is_terminal = np.array([n.is_terminal for n in tree.nodes], dtype=bool)
            is_success = np.array([n.is_success for n in tree.nodes], dtype=bool)

            meta = {
                "problem": problem,
                "n": tree.n,
                "parents": parents,
                "depths": depths,
                "is_terminal": is_terminal,
                "is_success": is_success,
            }
            torch.save(meta, meta_path)
            np.save(hidden_path, hidden)
            total_nodes += tree.n

            if (idx + 1) % 25 == 0 or idx == len(problems) - 1:
                elapsed = time.time() - start
                rate = (idx + 1) / max(elapsed, 1e-6)
                eta = (len(problems) - (idx + 1)) / rate
                print(f"  [{split}] {idx+1}/{len(problems)} | "
                      f"nodes={total_nodes} | {rate:.2f} prob/s | "
                      f"eta={eta/60:.1f} min", flush=True)

        print(f"[{split}] done: {total_nodes} total nodes", flush=True)


if __name__ == "__main__":
    main()
