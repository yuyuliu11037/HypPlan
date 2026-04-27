"""Import the public ProofWriter CWA depth-3 release into Group B JSONL.

Reads from
  external/proofwriter/proofwriter-dataset-V2020.12.3/CWA/depth-3/meta-{train,dev,test}.jsonl
and emits per-question records to
  data/proofwriter_{train,val,test}.jsonl

Each ProofWriter "theory" record contains 16 questions; we expand them to
16 separate per-question records.

Filtering rules (matches the dataset_construction.md spec):
- Use CWA depth-3 only.
- Mix of QDep ∈ {0,1,2,3} on test/val (gives an internal difficulty
  gradient).
- Train: include only QDep > 0 records (so we have a non-trivial proof
  chain to use as gold derivation for SFT-PT and DAgger training).

Schema per emitted line:
  {
    "id": "proofwriter_{split}_{i}",
    "theory_text": str,
    "initial_facts": [[S, V, O, "+"|"~"], ...],
    "rule_texts": {rule_id: str, ...},
    "rules_struct": {rule_id: {"premises": [...], "conclusion": [...]}, ...},
    "triple_texts": [[[S,V,O,p], "<NL text>"], ...],
    "target": [S, V, O, "+"|"~"],
    "target_text": str,
    "answer": bool,
    "QDep": int,
    "proof_chain": [{"rule_id": str, "intermediate": [S,V,O,p],
                       "intermediate_text": str}, ...],
    "prompt": str,
    "init_state_text": str,
    "answer_label": str,
  }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_proofwriter import (
    Problem, format_gold_trajectory, format_question, parse_rule, parse_triple,
    render_state,
)


def _intermediate_id_to_triple(int_block) -> dict:
    """Map intermediate's id -> {"text", "triple"}.

    `int_block` is a dict for QDep>0 records but can be an empty list `[]`
    for QDep=0 / no-intermediates records. Treat the list case as empty."""
    if not isinstance(int_block, dict):
        return {}
    out = {}
    for k, v in int_block.items():
        t = parse_triple(v["representation"])
        if t is not None:
            out[k] = {"text": v["text"], "triple": t}
    return out


# In the proofsWithIntermediates representation, each step like
# "(((triple7 triple1) -> (rule1 % int3)))" embeds the rule_id and the
# resulting intermediate id. We extract rule_id+int_id pairs in the order
# they appear left-to-right, but a SET of pairs (since the same step can
# appear multiple times in the proof DAG).
_STEP_PAIR_RE = re.compile(r"\((rule\d+)\s*%\s*(int\d+)\)")


def extract_proof_chain(proof_with_int: dict) -> list[dict]:
    """Reconstruct the linear chain int_K -> int_{K-1} -> ... -> int_1
    from the proofsWithIntermediates structure (where int1 is the final
    answer)."""
    if not proof_with_int:
        return []
    rep = proof_with_int.get("representation", "")
    int_block = proof_with_int.get("intermediates", {})
    int_map = _intermediate_id_to_triple(int_block)
    if not int_map:
        return []
    pairs = _STEP_PAIR_RE.findall(rep)
    # Deduplicate while preserving order of FIRST occurrence of each
    # (rule_id, int_id).
    seen: set = set()
    unique_pairs: list[tuple[str, str]] = []
    for r_id, i_id in pairs:
        key = (r_id, i_id)
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(key)
    # Forward-chaining order: derive deepest intermediates first.
    # Highest-numbered int_id was derived first (smallest QDep), int1 is the
    # final answer (largest QDep). So we sort by int_id DESC.
    def _int_num(s: str) -> int:
        return int(s[3:])
    unique_pairs.sort(key=lambda rp: -_int_num(rp[1]))
    chain: list[dict] = []
    for r_id, i_id in unique_pairs:
        if i_id not in int_map:
            continue
        chain.append({
            "rule_id": r_id,
            "intermediate": list(int_map[i_id]["triple"]),
            "intermediate_text": int_map[i_id]["text"],
        })
    return chain


def convert_question(theory_rec: dict, q_id: str, q: dict, idx: int,
                       split: str) -> dict | None:
    """Turn one question into our standard JSONL record."""
    target = parse_triple(q["representation"])
    if target is None:
        return None

    initial_facts: list[list] = []
    triple_texts: dict[tuple, str] = {}
    for tid, t in theory_rec["triples"].items():
        triple = parse_triple(t["representation"])
        if triple is None:
            continue
        initial_facts.append(list(triple))
        triple_texts[triple] = t["text"]

    rule_texts: dict[str, str] = {}
    rules_struct: dict[str, dict] = {}
    for rid, r in theory_rec["rules"].items():
        rule_texts[rid] = r["text"]
        parsed = parse_rule(r["representation"])
        if parsed is not None:
            rules_struct[rid] = {
                "premises": [list(p) for p in parsed["premises"]],
                "conclusion": list(parsed["conclusion"]),
            }

    answer = bool(q["answer"])
    proofs = q.get("proofsWithIntermediates", [])
    proof_chain = extract_proof_chain(proofs[0]) if proofs else []
    # Annotate triple_texts with intermediate texts too so render_state can
    # look them up.
    for step in proof_chain:
        triple = tuple(step["intermediate"])
        triple_texts[triple] = step["intermediate_text"]

    # Build a Problem so we can use the standard renderers.
    problem = Problem(
        theory_text=theory_rec["theory"],
        initial_facts=tuple(tuple(t) for t in initial_facts),
        rule_texts=rule_texts,
        rules_struct=rules_struct,
        triple_texts=triple_texts,
        target=tuple(target),
        target_text=q["question"],
        answer=answer,
        proof_chain=tuple({
            "rule_id": s["rule_id"],
            "intermediate": tuple(s["intermediate"]),
            "intermediate_text": s["intermediate_text"],
        } for s in proof_chain),
    )

    return {
        "id": f"proofwriter_{split}_{idx}",
        "theory_text": theory_rec["theory"],
        "initial_facts": initial_facts,
        "rule_texts": rule_texts,
        "rules_struct": rules_struct,
        "triple_texts": [
            [list(k), v] for k, v in triple_texts.items()
        ],
        "target": list(target),
        "target_text": q["question"],
        "answer": answer,
        "QDep": int(q["QDep"]),
        "proof_chain": proof_chain,
        "prompt": format_question(problem),
        "init_state_text": theory_rec["theory"],
        "answer_label": format_gold_trajectory(problem),
    }


def import_split(in_path: Path, out_path: Path, split: str,
                  max_records: int = -1, train_only_proofs: bool = False,
                  qdep_min: int = 0, qdep_max: int = 3) -> int:
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            theory_rec = json.loads(line)
            for q_id, q in theory_rec["questions"].items():
                qdep = int(q.get("QDep", 0))
                if qdep < qdep_min or qdep > qdep_max:
                    continue
                if train_only_proofs and not bool(q.get("answer", False)):
                    continue
                if train_only_proofs and qdep == 0:
                    continue
                rec = convert_question(theory_rec, q_id, q, n, split)
                if rec is None:
                    continue
                fout.write(json.dumps(rec) + "\n")
                n += 1
                if max_records > 0 and n >= max_records:
                    return n
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        default="external/proofwriter/proofwriter-dataset-V2020.12.3/CWA/depth-3",
    )
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_train", type=int, default=2000,
                     help="Train: True-answer + QDep>0 only")
    ap.add_argument("--n_val", type=int, default=200,
                     help="Val: full mix QDep ∈ [0,3] inc. False")
    ap.add_argument("--n_test", type=int, default=200,
                     help="Test: full mix QDep ∈ [0,3] inc. False")
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_train = import_split(
        src / "meta-train.jsonl", out / "proofwriter_train.jsonl",
        "train", max_records=args.n_train, train_only_proofs=True,
        qdep_min=1, qdep_max=3,
    )
    print(f"  wrote {n_train} -> proofwriter_train.jsonl")
    n_val = import_split(
        src / "meta-dev.jsonl", out / "proofwriter_val.jsonl",
        "val", max_records=args.n_val, train_only_proofs=False,
        qdep_min=0, qdep_max=3,
    )
    print(f"  wrote {n_val} -> proofwriter_val.jsonl")
    n_test = import_split(
        src / "meta-test.jsonl", out / "proofwriter_test.jsonl",
        "test", max_records=args.n_test, train_only_proofs=False,
        qdep_min=0, qdep_max=3,
    )
    print(f"  wrote {n_test} -> proofwriter_test.jsonl")


if __name__ == "__main__":
    main()
