"""Unified tree-data cache generator for Group B Stage-1 head training.

Produces the same metadata + hidden-state caches that
`generate_tree_data_pronto.py` etc. do, but dispatches on `--task` to the
right oracle. JSONL records are read from `data/{task}_{split}.jsonl`
(produced by the per-task generators in this directory) — the generator
parses each record, enumerates the oracle tree, renders each state, and
forwards through the frozen base model.

Output layout (matches Group A convention):
  data/{task}_trees_{model_tag}/
    {split}/
      problem_{idx}.pt        # metadata (n, parents, depths, is_terminal,
                              #            is_success, v_values, problem)
      hidden_{idx}.npy        # float16 (n, hidden_size) last-token hidden states

Usage (one rank per GPU; gloo not needed since this is pure forward-only):
  CUDA_VISIBLE_DEVICES=1 python -m data.generate_tree_data_groupB \\
    --task rulechain --splits train,val,test --shard_rank 0 --shard_world 6
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
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_problem_and_render(task: str, rec: dict):
    """Return (problem, tree, state_text_list) for one record."""
    if task in ("rulechain", "synthlogic"):
        from src.oracle_rulechain import (
            Problem, Rule, enumerate_tree, render_state,
        )
        rules = tuple(
            Rule(premises=tuple(sorted(r["premises"])),
                 conclusion=r["conclusion"])
            for r in rec["rules"]
        )
        p = Problem(
            initial_facts=frozenset(rec["initial_facts"]),
            target=rec["target"],
            rules=rules,
        )
        # Cap depth slightly above the gold derivation so trees stay small.
        max_depth = int(rec.get("n_steps", 4)) + 4
        tree = enumerate_tree(p, max_nodes=4000, max_depth=max_depth)
        texts = [render_state(p, n.state) for n in tree.nodes]
        return p, tree, texts

    if task == "clutrr":
        from src.oracle_clutrr import (
            Problem, enumerate_tree, render_state,
        )
        p = Problem(
            entities=tuple(rec["entities"]),
            edges=tuple((i, rel, j) for (i, rel, j) in rec["edges"]),
            query=tuple(rec["query"]),
            answer=rec["answer"],
            chain=tuple(rec["chain"]),
        )
        tree = enumerate_tree(p, max_nodes=200)
        texts = [render_state(p, n.state) for n in tree.nodes]
        return p, tree, texts

    if task == "lineq":
        from src.oracle_lineq import (
            Problem, State, enumerate_tree, render_state,
        )
        init = rec["initial"]
        p = Problem(
            initial=State(
                lhs_x=tuple(init["lhs_x"]),
                lhs_c=tuple(init["lhs_c"]),
                rhs_x=tuple(init["rhs_x"]),
                rhs_c=tuple(init["rhs_c"]),
            ),
            solution=int(rec["solution"]),
        )
        tree = enumerate_tree(p, max_nodes=600, max_depth=int(rec["k"]) + 2)
        texts = [render_state(p, n.state) for n in tree.nodes]
        return p, tree, texts

    if task == "proofwriter":
        from src.oracle_proofwriter import (
            Problem, enumerate_tree, render_state,
        )
        triple_texts = {tuple(k): v for (k, v) in rec.get("triple_texts", [])}
        p = Problem(
            theory_text=rec["theory_text"],
            initial_facts=tuple(tuple(t) for t in rec["initial_facts"]),
            rule_texts=dict(rec["rule_texts"]),
            rules_struct=dict(rec["rules_struct"]),
            triple_texts=triple_texts,
            target=tuple(rec["target"]),
            target_text=rec["target_text"],
            answer=bool(rec["answer"]),
            proof_chain=tuple({
                "rule_id": s["rule_id"],
                "intermediate": tuple(s["intermediate"]),
                "intermediate_text": s["intermediate_text"],
            } for s in rec["proof_chain"]),
        )
        tree = enumerate_tree(p)
        texts = [render_state(p, n.state) for n in tree.nodes]
        return p, tree, texts

    raise ValueError(f"Unknown task: {task}")


def _is_success(task: str, node) -> bool:
    """Map task-specific terminal-success predicate to the unified field."""
    if task in ("rulechain", "synthlogic", "lineq", "proofwriter"):
        return node.is_solved
    if task == "clutrr":
        return node.is_solved
    raise ValueError(task)


def build_one(task: str, rec: dict, model, tokenizer, device,
              max_length: int = 512) -> tuple[dict, np.ndarray]:
    p, tree, state_texts = _build_problem_and_render(task, rec)
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
        is_success[node.node_id] = _is_success(task, node)
        v_values[node.node_id] = node.v_value

    hiddens = np.zeros((n, model.config.hidden_size), dtype=np.float16)
    BATCH = 8
    with torch.no_grad():
        for s in range(0, n, BATCH):
            batch = state_texts[s : s + BATCH]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(device)
            out = model(**enc, output_hidden_states=False)
            last_h = out.last_hidden_state
            attn = enc["attention_mask"].sum(-1) - 1
            for bi in range(last_h.size(0)):
                hiddens[s + bi] = (
                    last_h[bi, int(attn[bi]), :].float().cpu().numpy()
                )

    meta = {
        "n": n, "parents": parents, "depths": depths,
        "is_terminal": is_terminal, "is_success": is_success,
        "v_values": v_values,
        "problem": rec.get("id", "unknown"),
    }
    return meta, hiddens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                     choices=["rulechain", "synthlogic", "clutrr", "lineq",
                              "proofwriter"])
    ap.add_argument("--data_prefix", default=None,
                     help="Override default data/{task}. Used for reading "
                          "data/{prefix}_{split}.jsonl instead.")
    ap.add_argument("--out_dir", default=None,
                     help="Override default data/{task}_trees_qwen14b")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    data_prefix = args.data_prefix or args.task
    out_dir = Path(args.out_dir or f"data/{args.task}_trees_qwen14b")
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

    for split_name in args.splits.split(","):
        path = Path(f"data/{data_prefix}_{split_name}.jsonl")
        if not path.exists():
            print(f"[r{args.shard_rank}] skip {split_name}: {path} missing",
                  flush=True)
            continue
        records = [json.loads(l) for l in open(path)]
        if args.limit > 0:
            records = records[: args.limit]
        shard = [(i, r) for (i, r) in enumerate(records)
                 if i % args.shard_world == args.shard_rank]
        print(f"[r{args.shard_rank}] {split_name}: {len(shard)} of "
              f"{len(records)} records", flush=True)
        for ti, (orig_idx, rec) in enumerate(shard):
            out_pt = out_dir / split_name / f"problem_{orig_idx}.pt"
            out_npy = out_dir / split_name / f"hidden_{orig_idx}.npy"
            if out_pt.exists() and out_npy.exists():
                continue
            t0 = time.time()
            meta, hiddens = build_one(args.task, rec, model, tokenizer,
                                       device, args.max_length)
            torch.save(meta, out_pt)
            np.save(out_npy, hiddens)
            if (ti + 1) % 25 == 0:
                print(f"  [r{args.shard_rank}/{split_name}] "
                      f"{ti+1}/{len(shard)} problem_{orig_idx}: "
                      f"{meta['n']} nodes in {time.time()-t0:.1f}s",
                      flush=True)


if __name__ == "__main__":
    main()
