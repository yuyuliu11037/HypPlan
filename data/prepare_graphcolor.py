"""Generate graph 3-coloring problems and SFT/eval datasets.

Produces:
  data/graphcolor_test.jsonl       — eval test set (200 problems, with fewshot prompt)
  data/graphcolor_train_sft_plan.jsonl — PT-SFT training data (250, planning-token augmented)
  data/graphcolor_problems.json    — raw problem list shared across head training
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_graphcolor import (
    generate_problem, format_question, format_gold_trajectory,
    Problem, COLOR_NAMES,
)


N_TRAIN, N_VAL, N_TEST = 250, 50, 200
TOTAL = N_TRAIN + N_VAL + N_TEST


# Two-shot exemplars used in eval prompts.
_FEWSHOT_PROBLEMS: list[tuple[Problem, tuple[str, ...]]] = [
    (Problem(n=4, edges=((0, 1), (1, 2), (2, 3)),
              one_solution=("R", "G", "R", "G")),
     ("R", "G", "R", "G")),
    (Problem(n=5, edges=((0, 1), (0, 2), (1, 2), (2, 3), (3, 4)),
              one_solution=("R", "G", "B", "R", "G")),
     ("R", "G", "B", "R", "G")),
]


def fewshot_eval_prompt(problem: Problem) -> str:
    parts = [
        "You are given a graph 3-coloring task. Output one assignment per "
        "line in the form 'V<i> = <color>' with color in {R, G, B}, then "
        "stop.",
        "",
    ]
    for ex_p, ex_sol in _FEWSHOT_PROBLEMS:
        parts.append(format_question(ex_p))
        parts.append("Coloring:")
        parts.append(format_gold_trajectory(ex_p, ex_sol,
                                              with_planning_tokens=False))
        parts.append("")
    parts.append(format_question(problem))
    parts.append("Coloring:")
    return "\n".join(parts)


def make_problem_pool(seed: int = 1234) -> list[Problem]:
    rng = random.Random(seed)
    pool: list[Problem] = []
    # Stratified across (n, density) — equal share per cell
    cells = [(n, d) for n in (5, 6) for d in (0.2, 0.3, 0.4, 0.5, 0.6)]
    per_cell = TOTAL // len(cells) + 1
    for n, d in cells:
        for _ in range(per_cell):
            p = generate_problem(n, d, rng)
            pool.append(p)
    rng.shuffle(pool)
    return pool[:TOTAL]


def main():
    out = Path("data")
    pool = make_problem_pool()
    train = pool[:N_TRAIN]
    val = pool[N_TRAIN:N_TRAIN + N_VAL]
    test = pool[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]

    # Test set: full eval prompts
    with (out / "graphcolor_test.jsonl").open("w") as f:
        for i, p in enumerate(test):
            init_state_text = format_question(p)   # used by HypPlan task-z to compute z
            f.write(json.dumps({
                "id": f"gc_{i}",
                "n": p.n, "edges": list(p.edges),
                "prompt": fewshot_eval_prompt(p),
                "init_state_text": init_state_text,
                "gold_solution": list(p.one_solution),
            }) + "\n")

    # PT-SFT training: question = format_question, answer = planning-token trajectory
    with (out / "graphcolor_train_sft_plan.jsonl").open("w") as f:
        for i, p in enumerate(train):
            q = format_question(p)
            a = format_gold_trajectory(p, p.one_solution,
                                          with_planning_tokens=True)
            f.write(json.dumps({
                "question": q, "answer": a,
                "n": p.n, "edges": list(p.edges),
                "gold_solution": list(p.one_solution),
            }) + "\n")

    # Raw problem pool for head training (train/val/test indices same as above)
    raw = []
    for split, items in [("train", train), ("val", val), ("test", test)]:
        for i, p in enumerate(items):
            raw.append({"split": split, "idx": i, "n": p.n,
                         "edges": list(p.edges),
                         "gold": list(p.one_solution)})
    with (out / "graphcolor_problems.json").open("w") as f:
        json.dump(raw, f)

    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print(f"Wrote: graphcolor_test.jsonl, graphcolor_train_sft_plan.jsonl, "
           f"graphcolor_problems.json")
    # Print a sample
    print("\nSample test prompt (truncated):")
    print(fewshot_eval_prompt(test[0])[:600])
    print("\nSample SFT answer:")
    print(format_gold_trajectory(train[0], train[0].one_solution,
                                    with_planning_tokens=True))


if __name__ == "__main__":
    main()
