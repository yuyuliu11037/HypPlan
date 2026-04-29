"""Re-score ProofWriter baselines + HypPlan on the multi-hop subset
(QDep >= 1).

Reads existing per-record result files for each method, joins with the
test set to get QDep when missing, parses generations for the final
answer when needed, then reports per-QDep and overall accuracy on the
QDep>=1 subset.

Output: prints a per-method table to stdout. Saves a CSV+JSONL summary.

Usage:
  python3.10 -m src.rescore_proofwriter_multihop
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


def parse_pw_answer(gen: str):
    """Return True/False/None from a ProofWriter generation."""
    if not gen:
        return None
    m = re.search(r"Answer\s*[:\-]?\s*(True|False)", gen, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def load_test_qdep(test_path: Path) -> dict[str, int]:
    qd = {}
    for line in open(test_path):
        r = json.loads(line)
        qd[r["id"]] = r["QDep"]
    return qd


def score_records(records, scorer, qdep_map, gold_map):
    """Apply `scorer(rec) -> bool|None` to each, return per-QDep dict
    {qd: [total, correct]}."""
    by_qd = defaultdict(lambda: [0, 0])
    for r in records:
        qd = r.get("QDep")
        if qd is None:
            qd = qdep_map.get(r["id"])
        if qd is None:
            continue
        ok = scorer(r)
        by_qd[qd][0] += 1
        if ok:
            by_qd[qd][1] += 1
    return dict(by_qd)


def report(name, by_qd):
    total = sum(t for t, c in by_qd.values())
    total_correct = sum(c for t, c in by_qd.values())
    multihop = {k: v for k, v in by_qd.items() if k >= 1}
    mh_total = sum(t for t, c in multihop.values())
    mh_correct = sum(c for t, c in multihop.values())

    print(f"\n=== {name} ===")
    print(f"  Full set:    {total_correct}/{total} = "
          f"{total_correct/total:.1%}" if total else "  (no records)")
    print(f"  Multi-hop:   {mh_correct}/{mh_total} = "
          f"{mh_correct/mh_total:.1%}" if mh_total else "  multi-hop empty")
    print(f"  Per-QDep:")
    for k in sorted(by_qd):
        t, c = by_qd[k]
        marker = "" if k >= 1 else "  (excluded)"
        print(f"    QDep={k}: {c}/{t} = {c/t:.1%}{marker}")
    return {
        "name": name,
        "full_total": total, "full_correct": total_correct,
        "mh_total": mh_total, "mh_correct": mh_correct,
        "per_qdep": {str(k): list(v) for k, v in by_qd.items()},
    }


def main():
    test_path = Path("data/proofwriter_test.jsonl")
    qdep_map = load_test_qdep(test_path)
    gold_map = {r["id"]: r["answer"] for r in
                (json.loads(l) for l in open(test_path))}
    print(f"Loaded {len(qdep_map)} test records")

    summaries = []

    # 1. Base 4-bit greedy
    f = Path("results/eval_groupB_base/proofwriter_base_4bit.jsonl")
    if f.exists():
        recs = [json.loads(l) for l in open(f)]
        s = score_records(recs, lambda r: bool(r.get("correct")),
                          qdep_map, gold_map)
        summaries.append(report(f"Base Qwen-14B-Instruct 4bit (greedy) [{f.name}]",
                                s))

    # 2. SC (majority vote)
    f = Path("results/baselines/proofwriter_sc.jsonl")
    if f.exists():
        recs = [json.loads(l) for l in open(f)]
        s = score_records(recs, lambda r: bool(r.get("majority_ok")),
                          qdep_map, gold_map)
        summaries.append(report(f"Self-Consistency (majority of K=5) [{f.name}]",
                                s))

    # 3. PT-SFT (parse generation)
    f = Path("results/baselines/proofwriter_ptsft.jsonl")
    if f.exists():
        recs = [json.loads(l) for l in open(f)]
        def pt_scorer(r):
            parsed = parse_pw_answer(r.get("generation", ""))
            if parsed is None:
                return False
            return bool(parsed) == bool(r.get("answer"))
        s = score_records(recs, pt_scorer, qdep_map, gold_map)
        summaries.append(report(f"Planning-Token SFT [{f.name}]", s))

    # 4. HypPlan answer-accuracy (new eval)
    f = Path("results/eval_stage2_indomain/proofwriter/proofwriter_answer.jsonl")
    if f.exists():
        recs = [json.loads(l) for l in open(f)]
        s = score_records(recs, lambda r: bool(r.get("correct")),
                          qdep_map, gold_map)
        summaries.append(report(f"HypPlan in-domain (answer-accuracy) [{f.name}]",
                                s))

    out = Path("results/proofwriter_multihop_summary.jsonl")
    with open(out, "w") as fout:
        for s in summaries:
            fout.write(json.dumps(s) + "\n")
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
