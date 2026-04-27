"""Offline tree-data generator for Countdown stage-1.

For each problem in data/cd_{split}.jsonl, build a sampled history-subtree
(mix of oracle-guided + random trajectories), render each node's canonical
state text, and cache the frozen Countdown SFT LLM's last-token last-layer
hidden state for every node.

Outputs per split under data/cd_trees/{split}/:
  problem_{idx}.pt    -- torch pickle of {"pool", "target", "n", "parents",
                          "depths", "is_terminal", "is_success", "v_values"}
  hidden_{idx}.npy    -- float16 array of shape (n, hidden_dim)

Distances are not stored; derive from parents+depths via pair_distances_lca.
Texts are not stored; re-render from the tree when needed.

Resume-safe: skips problems whose both output files already exist.
Shard-ready: pass --shard_rank/--shard_world to split across processes/GPUs.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tree_data_cd import sample_tree, render_state


def load_problems(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def encode_texts(texts: list[str], tokenizer, model, device,
                 batch_size: int) -> np.ndarray:
    hidden_dim = model.config.hidden_size
    out = np.empty((len(texts), hidden_dim), dtype=np.float16)
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt", add_special_tokens=True,
        ).to(device)
        outputs = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        B = last_hidden.size(0)
        row = last_hidden[torch.arange(B, device=device), last_idx]
        out[i: i + B] = row.float().cpu().numpy().astype(np.float16)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft_cd_merged")
    parser.add_argument("--out_dir", default="data/cd_trees")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cache_dir", default="data/cd_oracle_cache")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--n_trajectories", type=int, default=200)
    parser.add_argument("--n_guided", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--shard_world", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map={"": device},
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"hidden_dim = {hidden_dim}", flush=True)

    for split in args.splits:
        jsonl = Path(args.data_dir) / f"cd_{split}.jsonl"
        cache_file = Path(args.cache_dir) / f"{split}.pkl"
        if not (jsonl.exists() and cache_file.exists()):
            print(f"[{split}] skip — missing {jsonl} or {cache_file}", flush=True)
            continue

        problems = load_problems(str(jsonl))
        with cache_file.open("rb") as f:
            caches = pickle.load(f)
        if args.limit > 0:
            problems = problems[: args.limit]
            caches = caches[: args.limit]

        out_split = Path(args.out_dir) / split
        out_split.mkdir(parents=True, exist_ok=True)

        my_count = sum(1 for i in range(len(problems))
                       if i % args.shard_world == args.shard_rank)
        print(f"\n[{split}] {len(problems)} problems total, {my_count} this "
              f"shard (rank {args.shard_rank}/{args.shard_world})", flush=True)

        total_nodes = 0
        start = time.time()
        processed = 0
        for idx, (p, memo) in enumerate(zip(problems, caches)):
            if idx % args.shard_world != args.shard_rank:
                continue

            meta_path = out_split / f"problem_{idx}.pt"
            hidden_path = out_split / f"hidden_{idx}.npy"
            if meta_path.exists() and hidden_path.exists():
                total_nodes += torch.load(meta_path, weights_only=False)["n"]
                processed += 1
                continue

            tree = sample_tree(p["pool"], p["target"], memo,
                               n_trajectories=args.n_trajectories,
                               n_guided=args.n_guided,
                               seed=p["problem_idx"])
            texts = [render_state(tree, n) for n in tree.nodes]
            hidden = encode_texts(texts, tokenizer, model, device, args.batch_size)

            parents = np.array(
                [n.parent if n.parent is not None else -1 for n in tree.nodes],
                dtype=np.int32,
            )
            depths = np.array([n.depth for n in tree.nodes], dtype=np.int16)
            is_terminal = np.array([n.is_terminal for n in tree.nodes], dtype=bool)
            is_success = np.array([n.is_success for n in tree.nodes], dtype=bool)
            v_values = np.array([n.v_value for n in tree.nodes], dtype=np.int32)

            meta = {
                "pool": p["pool"], "target": p["target"],
                "n": tree.n,
                "parents": parents, "depths": depths,
                "is_terminal": is_terminal, "is_success": is_success,
                "v_values": v_values,
            }
            torch.save(meta, meta_path)
            np.save(hidden_path, hidden)
            total_nodes += tree.n
            processed += 1

            if processed % 25 == 0 or idx == len(problems) - 1:
                elapsed = time.time() - start
                rate = processed / max(elapsed, 1e-6)
                eta = (my_count - processed) / max(rate, 1e-6)
                print(f"  [{split}] rank {args.shard_rank}: "
                      f"{processed}/{my_count} | nodes={total_nodes} | "
                      f"{rate:.2f} prob/s | eta={eta/60:.1f} min", flush=True)

        print(f"[{split}] rank {args.shard_rank} done: {total_nodes} total nodes",
              flush=True)


if __name__ == "__main__":
    main()
