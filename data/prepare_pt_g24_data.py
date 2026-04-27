"""Reformat 24_train_sft3k_plan.jsonl ({problem, text}) into Question/Answer
format ({question, answer}) consistent with the OOD PT-SFT trainer
(train_sft_gsm8k.py).

The input `text` field is "Problem: <pool>\n<PLAN:OP> Step 1: ...". We split
on the first newline so question="Problem: <pool>" and answer="<PLAN:OP> ...".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/24_train_sft3k_plan.jsonl")
    ap.add_argument("--output", default="data/24_train_sft_pt.jsonl")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"]
            if "\n" not in text:
                continue
            q, a = text.split("\n", 1)
            fout.write(json.dumps({
                "question": q,
                "answer": a,
                "problem": rec["problem"],
            }) + "\n")
            n += 1
    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()
