"""Generate CLUTRR v4 with TEMPLATE-DISJOINT test split.

Differences from v3:
- Test contains chain templates that NEVER appear in train (true generalization)
- Test mix is multi-k (k=2,3,4) instead of k=4 only
- Val is smaller (saves head training time)
- Train/val use the train-template pool; test uses the held-out template pool

Pipeline:
  1. Sample N_enum candidate problems per k → collect unique chain templates
  2. Per-k template partition: shuffle with fixed seed, take last
     `--test_template_frac` for TEST (held out), rest for TRAIN
  3. Generate TRAIN/VAL: regenerate problems, ACCEPT only chains in train pool
  4. Generate TEST: regenerate problems, ACCEPT only chains in test pool;
     use a different seed_base so entity assignments are fresh

Schema is identical to v3 (same fields).

Usage:
  python3.10 data/generate_data_clutrr_v4.py \\
    --n_train 3000 --n_val 60 --n_test_per_k 30 \\
    --ks 2,3,4 \\
    --test_template_frac 0.25 \\
    --seed_base 9101
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_clutrr import (Problem, format_gold_trajectory,
                                 generate_problem, render_state)


def enumerate_chains(k: int, n_enum: int, seed_base: int,
                      n_distractor_entities: int,
                      n_distractor_edges: int) -> dict:
    """Run generate_problem n_enum times at k and collect (chain → count)."""
    counts: Counter = Counter()
    for i in range(n_enum):
        seed = seed_base + i * 7919  # arbitrary stride
        try:
            p = generate_problem(
                k=k, seed=seed,
                n_distractor_entities=n_distractor_entities,
                n_distractor_edges=n_distractor_edges,
            )
            counts[tuple(p.chain)] += 1
        except Exception:
            continue
    return counts


def write_jsonl(records: list[dict], path: Path, prefix: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, r in enumerate(records):
            r["id"] = f"clutrr_{prefix}_{i}"
            f.write(json.dumps(r) + "\n")


def render_record(p: Problem, k: int, split: str) -> dict:
    prompt_text = p.render_problem()
    gold = format_gold_trajectory(p)
    return {
        "k": k,
        "entities": list(p.entities),
        "edges": [list(e) for e in p.edges],
        "query": list(p.query),
        "answer": p.answer,
        "chain": list(p.chain),
        "prompt": prompt_text,
        "init_state_text": prompt_text,
        "answer_label": gold,
        "split": split,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--n_test_per_k", type=int, default=30)
    ap.add_argument("--ks", default="2,3,4")
    ap.add_argument("--n_enum_per_k", type=int, default=2000,
                     help="how many candidate problems to sample to estimate "
                          "the chain-template universe per k")
    ap.add_argument("--test_template_frac", type=float, default=0.25,
                     help="fraction of unique chain templates per k to "
                          "reserve EXCLUSIVELY for test")
    ap.add_argument("--n_distractor_entities", type=int, default=2)
    ap.add_argument("--n_distractor_edges", type=int, default=2)
    ap.add_argument("--seed_base", type=int, default=9101)
    ap.add_argument("--out_train", default="data/clutrr_train.jsonl")
    ap.add_argument("--out_val", default="data/clutrr_val.jsonl")
    ap.add_argument("--out_test", default="data/clutrr_test.jsonl")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    print(f"=== Step 1: enumerate chains per k ===", flush=True)
    train_pool: dict[int, set] = {}
    test_pool: dict[int, set] = {}
    for k in ks:
        counts = enumerate_chains(
            k=k, n_enum=args.n_enum_per_k,
            seed_base=args.seed_base + k * 31337,
            n_distractor_entities=args.n_distractor_entities,
            n_distractor_edges=args.n_distractor_edges,
        )
        chains = list(counts.keys())
        rng_split = random.Random(args.seed_base + k * 7 + 99)
        rng_split.shuffle(chains)
        n_test_chains = max(2,
                              int(round(len(chains) * args.test_template_frac)))
        test_chains = set(chains[:n_test_chains])
        train_chains = set(chains[n_test_chains:])
        train_pool[k] = train_chains
        test_pool[k] = test_chains
        print(f"  k={k}: total {len(chains)} unique chains; "
                f"train pool {len(train_chains)} | "
                f"test pool {len(test_chains)} (held out)", flush=True)

    # Sanity: train_pool ∩ test_pool == empty per k (must hold by construction).
    for k in ks:
        overlap = train_pool[k] & test_pool[k]
        assert not overlap, f"k={k} pool overlap: {overlap}"

    print(f"\n=== Step 2: generate TRAIN ({args.n_train}, train pool) ===",
            flush=True)
    train_records = []
    n_per_k = args.n_train // len(ks)
    for k in ks:
        rng_seed = args.seed_base + 10_000 + k * 100  # train seed
        attempt = 0
        produced = 0
        while produced < n_per_k and attempt < n_per_k * 50:
            seed = rng_seed + attempt * 13
            attempt += 1
            try:
                p = generate_problem(
                    k=k, seed=seed,
                    n_distractor_entities=args.n_distractor_entities,
                    n_distractor_edges=args.n_distractor_edges,
                )
                if tuple(p.chain) not in train_pool[k]:
                    continue
                train_records.append(render_record(p, k, "train"))
                produced += 1
            except Exception as exc:
                # Surface unexpected errors instead of silently swallowing
                # them — that's how the v3 'initial_state' bug hid for a
                # while. Only suppress the well-known generator failure
                # modes (e.g. valid-problem search exhausted).
                if "no valid problem" not in str(exc).lower():
                    raise
                continue
        print(f"  k={k}: produced {produced}/{n_per_k} train records "
                f"in {attempt} attempts", flush=True)

    print(f"\n=== Step 3: generate VAL ({args.n_val}, train pool) ===",
            flush=True)
    val_records = []
    n_val_per_k = args.n_val // len(ks)
    for k in ks:
        rng_seed = args.seed_base + 50_000 + k * 100
        attempt = 0
        produced = 0
        while produced < n_val_per_k and attempt < n_val_per_k * 50:
            seed = rng_seed + attempt * 13
            attempt += 1
            try:
                p = generate_problem(
                    k=k, seed=seed,
                    n_distractor_entities=args.n_distractor_entities,
                    n_distractor_edges=args.n_distractor_edges,
                )
                if tuple(p.chain) not in train_pool[k]:
                    continue
                val_records.append(render_record(p, k, "val"))
                produced += 1
            except Exception as exc:
                # Surface unexpected errors instead of silently swallowing
                # them — that's how the v3 'initial_state' bug hid for a
                # while. Only suppress the well-known generator failure
                # modes (e.g. valid-problem search exhausted).
                if "no valid problem" not in str(exc).lower():
                    raise
                continue
        print(f"  k={k}: produced {produced}/{n_val_per_k} val records",
                flush=True)

    print(f"\n=== Step 4: generate TEST ({args.n_test_per_k} per k, "
            f"test pool with fresh seeds) ===", flush=True)
    test_records = []
    for k in ks:
        rng_seed = args.seed_base + 200_000 + k * 100   # different seed_base
        attempt = 0
        produced = 0
        while produced < args.n_test_per_k and attempt < args.n_test_per_k * 100:
            seed = rng_seed + attempt * 13
            attempt += 1
            try:
                p = generate_problem(
                    k=k, seed=seed,
                    n_distractor_entities=args.n_distractor_entities,
                    n_distractor_edges=args.n_distractor_edges,
                )
                if tuple(p.chain) not in test_pool[k]:
                    continue
                test_records.append(render_record(p, k, "test"))
                produced += 1
            except Exception as exc:
                # Surface unexpected errors instead of silently swallowing
                # them — that's how the v3 'initial_state' bug hid for a
                # while. Only suppress the well-known generator failure
                # modes (e.g. valid-problem search exhausted).
                if "no valid problem" not in str(exc).lower():
                    raise
                continue
        print(f"  k={k}: produced {produced}/{args.n_test_per_k} test records",
                flush=True)

    # Sanity: no train chain appears in test split, and vice versa.
    train_chains_seen = {(r["k"], tuple(r["chain"])) for r in train_records + val_records}
    test_chains_seen = {(r["k"], tuple(r["chain"])) for r in test_records}
    overlap = train_chains_seen & test_chains_seen
    assert not overlap, f"chain template overlap between train/val and test: {overlap}"
    print(f"\nVerified: 0 chain-template overlap between train/val and test.",
            flush=True)

    write_jsonl(train_records, Path(args.out_train), "train")
    write_jsonl(val_records, Path(args.out_val), "val")
    write_jsonl(test_records, Path(args.out_test), "test")
    print(f"\nWrote {len(train_records)}/{len(val_records)}/{len(test_records)} "
            f"records (train/val/test)", flush=True)
    print(f"Test k-distribution: "
            f"{Counter(r['k'] for r in test_records)}", flush=True)


if __name__ == "__main__":
    main()
