"""Generate CLUTRR-Graph train/test data for the PT-SFT memorization probe
and the eventual full pipeline.

Difference from gen_clutrr_graph_probe.py:
- Renders `answer_label` (the gold trajectory text) so the existing
  data/annotate_sft_plan_groupB.py script can add `<PLAN:*>` tags.
- Splits into train and test with template-disjoint chains (the test
  chain pattern never appears in train, mirroring v4's strategy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.oracle_clutrr import compose
from src.oracle_clutrr_graph import generate_graph_problem


def render_answer_label(p) -> str:
    """Walk gold_chain and gold_relations, emitting cumulative
    composition lines (matches chain CLUTRR convention)."""
    head = p.entities[p.gold_chain[0]]
    cur_rel = p.gold_relations[0]
    next_e = p.entities[p.gold_chain[1]]
    lines = [f"Step 1: {head} is the {cur_rel} of {next_e}"]
    for i in range(1, len(p.gold_relations)):
        cur_rel = compose(cur_rel, p.gold_relations[i])
        if cur_rel is None:
            return ""
        next_e = p.entities[p.gold_chain[i + 1]]
        lines.append(f"Step {i+1}: {head} is the {cur_rel} of {next_e}")
    tail = p.entities[p.gold_chain[-1]]
    lines.append(f"Answer: {head} is the {p.answer} of {tail}.")
    return "\n".join(lines)


def to_record(p, idx: int, split: str):
    story = p.render()
    answer_label = render_answer_label(p)
    return {
        "k": len(p.gold_relations),
        "entities": list(p.entities),
        "edges": [list(e) for e in p.edges],
        "query": list(p.query),
        "answer": p.answer,
        "chain": list(p.gold_relations),
        "prompt": story,
        "init_state_text": story,
        "answer_label": answer_label,
        "split": split,
        "id": f"clutrr_graph_{split}_{idx:05d}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--n_distractor_entities", type=int, default=12)
    ap.add_argument("--n_distractor_edges", type=int, default=20)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed_start", type=int, default=20000)
    ap.add_argument("--test_chain_frac", type=float, default=0.25,
                    help="Fraction of distinct chain templates reserved for test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: enumerate distinct chain templates by sampling problems.
    pool = []
    seen_chains = set()
    seed = args.seed_start
    target_pool = (args.n_train + args.n_test) * 4   # over-generate
    while len(pool) < target_pool and seed < args.seed_start + 100000:
        p = generate_graph_problem(
            k=args.k,
            n_distractor_entities=args.n_distractor_entities,
            n_distractor_edges=args.n_distractor_edges,
            seed=seed,
        )
        seed += 1
        if p is None:
            continue
        pool.append(p)
        seen_chains.add(p.gold_relations)

    chains = sorted(seen_chains)
    n_chains = len(chains)
    n_test_chains = max(1, int(args.test_chain_frac * n_chains))
    test_chains = set(chains[:n_test_chains])
    train_chains = set(chains[n_test_chains:])

    # Phase 2: partition pool by chain.
    train_buf = [p for p in pool if p.gold_relations in train_chains]
    test_buf = [p for p in pool if p.gold_relations in test_chains]
    train_buf = train_buf[:args.n_train]
    test_buf = test_buf[:args.n_test]

    print(f"[gen] distinct chains in pool: {n_chains} "
          f"(train={n_chains - n_test_chains}, test={n_test_chains})")
    print(f"[gen] train records: {len(train_buf)}, test records: {len(test_buf)}")

    if len(train_buf) < args.n_train:
        print(f"WARN: only generated {len(train_buf)} of {args.n_train} train")
    if len(test_buf) < args.n_test:
        print(f"WARN: only generated {len(test_buf)} of {args.n_test} test")

    train_path = out_dir / "clutrr_graph_train.jsonl"
    test_path = out_dir / "clutrr_graph_test.jsonl"
    with open(train_path, "w") as fout:
        for i, p in enumerate(train_buf):
            fout.write(json.dumps(to_record(p, i, "train")) + "\n")
    with open(test_path, "w") as fout:
        for i, p in enumerate(test_buf):
            fout.write(json.dumps(to_record(p, i, "test")) + "\n")
    print(f"wrote {train_path} ({len(train_buf)}) and {test_path} ({len(test_buf)})")
    print(f"[gen] verify split: shared chains = "
          f"{len(set(p.gold_relations for p in train_buf) & set(p.gold_relations for p in test_buf))}")


if __name__ == "__main__":
    main()
