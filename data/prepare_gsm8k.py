"""Download GSM8K and produce two SFT-ready jsonl variants.

Output:
  data/gsm8k_train_baseline.jsonl  — original answers, no planning tokens
  data/gsm8k_train_plan.jsonl      — <PLAN:OP> inserted before each step
                                     based on the operator in the <<a op b=r>>
                                     annotation. Multiple ops in one step → use
                                     the first; no annotation → <PLAN:OTHER>.
  data/gsm8k_test.jsonl            — test split with question + ground truth N

Each train record schema:
  {"question": "...", "answer": "<original or augmented>", "final": int}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset


EQ_RE = re.compile(r"<<([^=]+)=([^>]+)>>")
OP_RE = re.compile(r"([+\-*/])")
FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def first_op_from_step(step: str) -> str | None:
    """Get the operator from the first <<a op b = r>> annotation in step."""
    m = EQ_RE.search(step)
    if not m:
        return None
    expr = m.group(1)
    om = OP_RE.search(expr)
    return om.group(1) if om else None


def insert_planning_tokens(answer: str) -> str:
    """Prepend <PLAN:OP> (or <PLAN:OTHER>) before each step line.

    A 'step' is a line with at least one <<>> annotation. Lines without
    annotations (typically pure prose) get no tag. The final '#### N' line
    gets <PLAN:ANS>.
    """
    out_lines = []
    for line in answer.split("\n"):
        if line.startswith("####"):
            out_lines.append(f"<PLAN:ANS> {line}")
        elif EQ_RE.search(line):
            op = first_op_from_step(line) or "OTHER"
            out_lines.append(f"<PLAN:{op}> {line}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def extract_final(answer: str) -> int | float | None:
    m = FINAL_RE.search(answer)
    if not m:
        return None
    s = m.group(1)
    return float(s) if "." in s else int(s)


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print("Downloading GSM8K...", flush=True)
    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    print(f"  train={len(train)} test={len(test)}", flush=True)

    n_with_op = 0
    n_other = 0
    n_total_steps = 0
    with (out_dir / "gsm8k_train_baseline.jsonl").open("w") as f_base, \
         (out_dir / "gsm8k_train_plan.jsonl").open("w") as f_plan:
        for ex in train:
            q = ex["question"]
            a = ex["answer"]
            final = extract_final(a)
            f_base.write(json.dumps({"question": q, "answer": a,
                                       "final": final}) + "\n")
            a_plan = insert_planning_tokens(a)
            f_plan.write(json.dumps({"question": q, "answer": a_plan,
                                       "final": final}) + "\n")
            for line in a.split("\n"):
                if EQ_RE.search(line):
                    n_total_steps += 1
                    if first_op_from_step(line) is not None:
                        n_with_op += 1
                    else:
                        n_other += 1

    with (out_dir / "gsm8k_test.jsonl").open("w") as fout:
        for ex in test:
            fout.write(json.dumps({
                "question": ex["question"], "answer": ex["answer"],
                "final": extract_final(ex["answer"]),
            }) + "\n")

    print(f"\nTotal training steps with <<>>: {n_total_steps}")
    print(f"  with identifiable op: {n_with_op}")
    print(f"  fallback to OTHER:    {n_other}")
    print()
    # Sample
    print("Sample baseline:")
    print(json.loads(open(out_dir / "gsm8k_train_baseline.jsonl").readline())["answer"][:200])
    print()
    print("Sample plan:")
    print(json.loads(open(out_dir / "gsm8k_train_plan.jsonl").readline())["answer"][:300])


if __name__ == "__main__":
    main()
