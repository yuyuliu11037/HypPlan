"""Prepare ProntoQA + Blocksworld OOD test sets.

Output:
  data/prontoqa_test.jsonl   — 200 deductive reasoning Q&A
  data/blocksworld_test.jsonl — 200 natural-language planning problems

Schema for both:
  {"prompt": "<full-fewshot-prompt>", "answer_label": "<gold>"}
ProntoQA `answer_label` ∈ {"A", "B"} (corresponds to True/False).
Blocksworld `answer_label` is the action sequence (newline-separated).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


PRONTO_INSTRUCTION = (
    "You are given a set of facts and rules. Use them to determine whether "
    "the statement at the end is true or false. Answer with exactly one "
    "letter: 'A' for true, 'B' for false. Do not add any explanation."
)


def build_prontoqa_prompt(rec: dict, fewshot_examples: list[dict]) -> str:
    """Construct a fewshot prompt for ProntoQA."""
    parts = [PRONTO_INSTRUCTION, ""]
    for ex in fewshot_examples:
        parts.append(f"Context: {ex['context']}")
        parts.append(f"{ex['question']}")
        parts.append(f"{' '.join(ex['options'])}")
        parts.append(f"Answer: {ex['answer']}")
        parts.append("")
    parts.append(f"Context: {rec['context']}")
    parts.append(f"{rec['question']}")
    parts.append(f"{' '.join(rec['options'])}")
    parts.append("Answer:")
    return "\n".join(parts)


def main():
    out_dir = Path("data")
    rng = random.Random(1234)

    # ProntoQA: 500 records, only validation split
    print("Loading ProntoQA...", flush=True)
    pronto = load_dataset("renma/ProntoQA", split="validation")
    # Use first 3 as fewshot, next 200 as test
    # Use the head's seed-1234 split: test = records 300-499 (200) so the
    # head's training set (records 0-249) is disjoint from eval test.
    import sys
    sys.path.insert(0, str(out_dir.parent))
    from src.oracle_pronto import parse_problem, render_state
    rng2 = random.Random(1234)
    indices = list(range(len(pronto)))
    rng2.shuffle(indices)
    fewshot_indices = indices[:3]
    test_indices = indices[300:500]
    fewshot = [pronto[i] for i in fewshot_indices]
    test = [pronto[i] for i in test_indices]
    with (out_dir / "prontoqa_test.jsonl").open("w") as f:
        for rec in test:
            p = parse_problem(rec["raw_logic_programs"])
            init_state_text = render_state(p, p.facts)
            f.write(json.dumps({
                "id": rec["id"],
                "prompt": build_prontoqa_prompt(rec, fewshot),
                "answer_label": rec["answer"],  # "A" or "B"
                "init_state_text": init_state_text,
            }) + "\n")
    print(f"  ProntoQA: {len(test)} test records written "
          f"(disjoint from head train indices 0-249)")

    # Blocksworld via PlanBench task_1 plan_generation
    print("Loading PlanBench Blocksworld...", flush=True)
    bw = load_dataset("tasksource/planbench", "task_1_plan_generation",
                        split="train")
    bw_recs = [ex for ex in bw if ex["domain"] == "blocksworld"
               and ex["prompt_type"] == "oneshot"]
    print(f"  Found {len(bw_recs)} blocksworld oneshot records")
    rng.shuffle(bw_recs)
    # First 200 = test (matches head train script's split: 0-199 = test,
    # 200-449 = train, 450-499 = val).
    test_bw = bw_recs[:200]
    with (out_dir / "blocksworld_test.jsonl").open("w") as f:
        for rec in test_bw:
            f.write(json.dumps({
                "id": rec["instance_id"],
                "prompt": rec["query"],
                "answer_label": rec["ground_truth_plan"].strip(),
            }) + "\n")
    print(f"  Blocksworld: {len(test_bw)} test records written "
          f"(disjoint from head train indices 200-449)")


if __name__ == "__main__":
    main()
