"""Prototype probe for CLUTRR-Graph.

Purpose: verify before committing to the full pipeline that
1. the oracle generates valid graph instances with unique shortest gold path,
2. the dead-end ratio at the initial state is high (target ~85%),
3. the generated stories look reasonable.

Run:
    python3.10 -m scripts.probe_clutrr_graph \\
        --k 4 --n_distractor_entities 8 --n_distractor_edges 12 \\
        --n_instances 5 --seed 0
"""
from __future__ import annotations

import argparse
import json

from src.oracle_clutrr_graph import (
    generate_graph_problem,
    legal_compositions,
    progressing_compositions,
    _build_pair_relation_map,
)


def analyze(p, idx):
    print(f"\n========= INSTANCE {idx} =========")
    print(f"Entities ({len(p.entities)}): {list(p.entities)}")
    print(f"Edges ({len(p.edges)}):")
    for i, r, j in p.edges:
        print(f"  {p.entities[i]} is the {r} of {p.entities[j]}")
    qh, qt = p.query
    print(f"Query: how is {p.entities[qh]} related to {p.entities[qt]}?")
    print(f"Answer: {p.answer}")
    print(f"Gold chain (k={len(p.gold_relations)}): "
          + " → ".join(p.entities[e] for e in p.gold_chain))
    print(f"Gold relations: {list(p.gold_relations)}")

    rel_map = _build_pair_relation_map(p.edges)
    legal = legal_compositions(rel_map)
    prog = progressing_compositions(legal, p.query, rel_map)
    n_legal = len(legal)
    n_prog = len(prog)
    de_ratio = 1.0 - (n_prog / max(1, n_legal))
    print(f"\nLegal compositions at initial state: {n_legal}")
    print(f"Progressing (on gold chain): {n_prog}")
    print(f"Dead-end ratio: {de_ratio:.2%}")

    # Sample of distractor compositions
    distractors = [c for c in legal if c not in prog]
    if distractors:
        print(f"Sample distractor compositions:")
        for entry in distractors[:3]:
            (a, b), (b2, c), r1, r2, r3, ac = entry
            print(f"  ({p.entities[a]} {r1}-of {p.entities[b]}) ∘ "
                  f"({p.entities[b]} {r2}-of {p.entities[c]}) = "
                  f"({p.entities[a]} {r3}-of {p.entities[c]})")

    return n_legal, n_prog, de_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_distractor_entities", type=int, default=8)
    ap.add_argument("--n_distractor_edges", type=int, default=12)
    ap.add_argument("--n_instances", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--out_jsonl", default="")
    args = ap.parse_args()

    stats = []
    instances = []
    seed = args.seed_start
    while len(instances) < args.n_instances:
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

    for i, p in enumerate(instances):
        nl, np, der = analyze(p, i)
        stats.append((nl, np, der))

    print("\n========= SUMMARY =========")
    print(f"n_instances={len(instances)}, k={args.k}, "
          f"distractor_entities={args.n_distractor_entities}, "
          f"distractor_edges={args.n_distractor_edges}")
    avg_legal = sum(s[0] for s in stats) / len(stats)
    avg_prog = sum(s[1] for s in stats) / len(stats)
    avg_der = sum(s[2] for s in stats) / len(stats)
    print(f"avg legal compositions = {avg_legal:.1f}")
    print(f"avg progressing comps  = {avg_prog:.1f}")
    print(f"avg dead-end ratio     = {avg_der:.2%}")

    if args.out_jsonl:
        with open(args.out_jsonl, "w") as fout:
            for p in instances:
                fout.write(json.dumps({
                    "entities": list(p.entities),
                    "edges": [list(e) for e in p.edges],
                    "query": list(p.query),
                    "answer": p.answer,
                    "gold_chain": list(p.gold_chain),
                    "gold_relations": list(p.gold_relations),
                    "story": p.render(),
                }) + "\n")
        print(f"wrote {len(instances)} instances to {args.out_jsonl}")


if __name__ == "__main__":
    main()
