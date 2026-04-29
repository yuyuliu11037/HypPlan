"""Add planning-token annotations to Group B JSONL train splits.

Reads `data/{task}_train.jsonl` and writes `data/{task}_train_sft_plan.jsonl`
with the schema PT-SFT trainers expect:
  {"question": str, "answer": str, "id": str, ...}

The "answer" field is the gold trajectory rewritten with `<PLAN:...>`
prefixes per step + `<PLAN:ANS>` before the final answer line. Tag set is
task-specific:
  rulechain / synthlogic : <PLAN:APPLY> per step
  clutrr                  : <PLAN:COMPOSE> per step
  minisudoku              : <PLAN:PLACE>   per step

The "question" field is the prompt body (rec["prompt"]).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_PLAN_TAGS: dict[str, str] = {
    "rulechain": "<PLAN:APPLY>",
    "synthlogic": "<PLAN:APPLY>",
    "clutrr": "<PLAN:COMPOSE>",
    "lineq": "<PLAN:OP>",
    "proofwriter": "<PLAN:APPLY>",
    "numpath": "<PLAN:OP>",
    "nqueens": "<PLAN:PLACE>",
}


def _nqueens_prompt_and_answer(rec: dict) -> tuple[str, str]:
    """Build (prompt, answer_label) for an N-Queens record on the fly,
    since the universe-partition data generator does not include these
    fields. answer_label is one Step line per row plus the final
    'Solution: [...]' line."""
    from src.oracle_nqueens import Problem, format_question
    N = int(rec["N"])
    prefix = list(rec.get("prefix", []))
    gold = list(rec["gold_extension"])
    prob = Problem(N=N, prefix=tuple(prefix))
    prompt = format_question(prob)
    # Trajectory: emit one step line for each row from len(prefix)+1 onward.
    lines = []
    for r, c in enumerate(gold, 1):
        if r <= len(prefix):
            continue   # already placed in problem statement
        lines.append(f"Step {r}: Place queen in row {r} at column {c}.")
    sol_str = ", ".join(str(c) for c in gold)
    lines.append(f"Solution: [{sol_str}]")
    answer_label = "\n".join(lines)
    return prompt, answer_label


def annotate(answer_label: str, plan_tag: str) -> str:
    """Walk the gold trajectory text line-by-line, tag each `Step N:` line
    with `plan_tag`, tag the `Answer:`/`Solution:` line with `<PLAN:ANS>`."""
    out_lines: list[str] = []
    for ln in answer_label.split("\n"):
        s = ln.strip()
        if s.startswith("Step ") and ":" in s:
            out_lines.append(f"{plan_tag} {ln}")
        elif s.startswith("Answer:") or s.startswith("Solution:"):
            out_lines.append(f"<PLAN:ANS> {ln}")
        else:
            # multi-line answer body (e.g., minisudoku grid) — leave as-is
            out_lines.append(ln)
    return "\n".join(out_lines)


def convert_record(rec: dict, plan_tag: str, task: str) -> dict:
    if task == "nqueens":
        prompt, answer_label = _nqueens_prompt_and_answer(rec)
    else:
        prompt = rec["prompt"]
        answer_label = rec["answer_label"]
    return {
        "question": prompt,
        "answer": annotate(answer_label, plan_tag),
        "id": rec.get("id", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                     choices=list(_PLAN_TAGS.keys()))
    ap.add_argument("--in_path", default=None)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path or f"data/{args.task}_train.jsonl")
    out_path = Path(args.out_path or f"data/{args.task}_train_sft_plan.jsonl")
    plan_tag = _PLAN_TAGS[args.task]

    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for ln in fin:
            rec = json.loads(ln)
            out_rec = convert_record(rec, plan_tag, args.task)
            fout.write(json.dumps(out_rec) + "\n")
            n += 1
    print(f"wrote {n} records: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
