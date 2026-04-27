"""Convert G24 SFT trajectories to include planning token hints.

Implements the 'arithmetic' variant of planning tokens (Wang et al. 2023,
https://arxiv.org/abs/2310.05707) specialized to Game-of-24:
- K = 5 token types: <PLAN:+>, <PLAN:->, <PLAN:*>, <PLAN:/>, <PLAN:ANS>
- A planning token is inserted *before* each reasoning step, encoding the
  operator that step will use. The final "Answer: 24" line gets <PLAN:ANS>.

Input  schema (as in data/24_train_sft3k_tot.jsonl):
  {"problem": "1,2,3,4", "text": "Problem: 1 2 3 4\\nStep 1: 1 + 2 = 3. ...",
   "step_offsets": [...]}

Output schema (additional 'text' field with planning tokens):
  {"problem": "1,2,3,4", "text": "<PLAN:+> Step 1: 1 + 2 = 3. Remaining:...
                                   <PLAN:*> Step 3: ... <PLAN:ANS> Answer: 24"}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(
    r"Step\s+\d+:\s+\S+\s+([+\-*/])\s+\S+\s+=\s+\S+")


def insert_planning_tokens(text: str) -> str:
    """Insert <PLAN:OP> before each 'Step N:' and <PLAN:ANS> before 'Answer:'.

    If the last step already ends with 'Answer: X', the <PLAN:ANS> goes right
    before 'Answer:'. Otherwise we skip it (shouldn't happen with our data).
    """
    # Replace every 'Step N: a op b = r' with '<PLAN:op> Step N: ...'
    def replace_step(m: re.Match) -> str:
        op = m.group(1)
        # Insert before the matched step substring.
        return f"<PLAN:{op}> {m.group(0)}"

    # We want to prefix each Step occurrence. Use finditer + reconstruct so we
    # only prepend, not alter the inner text.
    out_parts = []
    last_end = 0
    for m in STEP_RE.finditer(text):
        out_parts.append(text[last_end : m.start()])
        out_parts.append(f"<PLAN:{m.group(1)}> ")
        out_parts.append(m.group(0))
        last_end = m.end()
    out_parts.append(text[last_end:])
    out = "".join(out_parts)

    # Insert <PLAN:ANS> before 'Answer:'
    out = out.replace("Answer:", "<PLAN:ANS> Answer:", 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="data/24_train_sft3k_tot.jsonl")
    ap.add_argument("--output",
                    default="data/24_train_sft_plan.jsonl")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    n = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            new_text = insert_planning_tokens(rec["text"])
            # Keep original fields + overwrite text
            rec["text"] = new_text
            # step_offsets are no longer accurate — drop them
            rec.pop("step_offsets", None)
            fout.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} records → {out_path}")
    # Print a sample
    print("\nSample:")
    with out_path.open() as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(json.loads(line)["text"][:200])
            print("---")


if __name__ == "__main__":
    main()
