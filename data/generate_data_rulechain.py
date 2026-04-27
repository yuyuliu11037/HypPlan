"""Generate synthetic rule-chaining JSONL data for Group B.

Creates train/val/test splits + balanced training set, mirroring
`24_varied_bal` for the Group A training source.

Output:
  data/rulechain_{train,val,test}.jsonl              raw, depth in {2,3,4}
  data/rulechain_bal_{train,val,test}.jsonl          balanced over depth (1/3 each)

Schema per line:
  {
    "id": str,
    "initial_facts": list[str],
    "target": str,
    "rules": list[{"premises": list[str], "conclusion": str}],
    "n_steps": int,                  # target_depth = oracle min derivation depth
    "n_predicates": int,
    "n_rules": int,
    "prompt": str,                   # full NL rendering of problem
    "init_state_text": str,          # initial state alone (for head input)
    "answer_label": str,             # gold trajectory text
    "split": str,
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_rulechain import (
    Problem, Rule, format_step_text, generate_problem, render_state,
)


def render_full_problem(p: Problem) -> str:
    return p.render_problem()


def render_init_state_text(p: Problem) -> str:
    """Render only the initial state for Stage-1 head input (matches the
    `init_state_text` field convention from the Group A test JSONLs)."""
    return render_state(p, p.initial_facts)


def render_gold_trajectory(p: Problem) -> str:
    """Walk one shortest derivation path and render as supervised target.
    Format mirrors `format_gold_trajectory` shape from oracle_clutrr."""
    from src.oracle_rulechain import _min_dist_to_target, applicable_rules, apply_rule
    state = p.initial_facts
    lines: list[str] = []
    step = 1
    safety = 16
    while p.target not in state and safety > 0:
        safety -= 1
        # pick any winning rule (greedy — gold is one valid path)
        from src.oracle_rulechain import winning_steps
        wins = winning_steps(state, p)
        if not wins:
            break
        r = wins[0]
        lines.append(f"Step {step}: {format_step_text(r)}")
        state = apply_rule(state, r)
        step += 1
    lines.append(f"Answer: {p.target} is derived.")
    return "\n".join(lines)


def problem_to_record(
    p: Problem, idx: int, depth: int, n_predicates: int, n_rules: int,
    split: str, task_name: str = "rulechain",
) -> dict:
    return {
        "id": f"{task_name}_{split}_{idx}",
        "initial_facts": sorted(p.initial_facts),
        "target": p.target,
        "rules": [
            {"premises": list(r.premises), "conclusion": r.conclusion}
            for r in p.rules
        ],
        "n_steps": depth,
        "n_predicates": n_predicates,
        "n_rules": n_rules,
        "prompt": render_full_problem(p),
        "init_state_text": render_init_state_text(p),
        "answer_label": render_gold_trajectory(p),
        "split": split,
    }


def gen_split(
    n: int, depths: list[int], n_predicates: int, n_rules: int,
    seed_offset: int, split: str, task_name: str = "rulechain",
    pred_prefix: str = "p",
) -> list[dict]:
    """Round-robin across `depths` to give a balanced distribution."""
    out: list[dict] = []
    di = 0
    seed = seed_offset
    while len(out) < n:
        depth = depths[di % len(depths)]
        di += 1
        try:
            p = generate_problem(
                n_predicates=n_predicates, n_rules=n_rules,
                target_depth=depth, seed=seed, pred_prefix=pred_prefix,
            )
            out.append(problem_to_record(
                p, len(out), depth, n_predicates, n_rules, split, task_name
            ))
        except RuntimeError:
            pass
        seed += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--depths", default="2,3,4",
                     help="Comma-separated training depth distribution")
    ap.add_argument("--n_predicates", type=int, default=16)
    ap.add_argument("--n_rules", type=int, default=18)
    ap.add_argument("--name_prefix", default="rulechain")
    ap.add_argument("--pred_prefix", default="p")
    ap.add_argument("--seed_base", type=int, default=1234)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.n_train, args.seed_base),
        ("val", args.n_val, args.seed_base + 100_000),
        ("test", args.n_test, args.seed_base + 200_000),
    ]
    for split, n, seed in splits:
        print(f"Generating {split}: n={n}, depths={depths}", flush=True)
        recs = gen_split(
            n=n, depths=depths,
            n_predicates=args.n_predicates, n_rules=args.n_rules,
            seed_offset=seed, split=split, task_name=args.name_prefix,
            pred_prefix=args.pred_prefix,
        )
        path = out_dir / f"{args.name_prefix}_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} records -> {path}", flush=True)


if __name__ == "__main__":
    main()
