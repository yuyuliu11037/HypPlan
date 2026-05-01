"""Generate a small CLUTRR-Graph probe set as JSONL using the same schema
as existing CLUTRR test data, so we can pipe it through the baseline
evaluator without modification.

Run:
    python3.10 -m scripts.gen_clutrr_graph_probe \\
        --k 4 --n_instances 20 \\
        --n_distractor_entities 12 --n_distractor_edges 20 \\
        --out data/clutrr_graph_probe.jsonl --seed_start 0
"""
from __future__ import annotations

import argparse
import json

from src.oracle_clutrr_graph import generate_graph_problem


def to_record(p, idx: int):
    """Match the schema of data/clutrr_test.jsonl:
    k, entities, edges, query, answer, chain, prompt, init_state_text,
    answer_label, split, id."""
    story = p.render()
    return {
        "k": len(p.gold_relations),
        "entities": list(p.entities),
        "edges": [list(e) for e in p.edges],
        "query": list(p.query),
        "answer": p.answer,
        "chain": list(p.gold_relations),
        "prompt": story,
        "init_state_text": story,
        "answer_label": p.answer,
        "split": "probe",
        "id": f"clutrr_graph_probe_{idx:04d}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_instances", type=int, default=20)
    ap.add_argument("--n_distractor_entities", type=int, default=12)
    ap.add_argument("--n_distractor_edges", type=int, default=20)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    instances = []
    seed = args.seed_start
    while len(instances) < args.n_instances and seed < args.seed_start + 5000:
        p = generate_graph_problem(
            k=args.k,
            n_distractor_entities=args.n_distractor_entities,
            n_distractor_edges=args.n_distractor_edges,
            seed=seed,
        )
        seed += 1
        if p is None:
            continue
        instances.append(p)

    if len(instances) < args.n_instances:
        print(f"WARN: only generated {len(instances)} of {args.n_instances}")

    with open(args.out, "w") as fout:
        for i, p in enumerate(instances):
            fout.write(json.dumps(to_record(p, i)) + "\n")
    print(f"wrote {len(instances)} records to {args.out}")


if __name__ == "__main__":
    main()
