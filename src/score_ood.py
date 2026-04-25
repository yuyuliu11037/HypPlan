"""Score OOD eval outputs.

ProntoQA: extract first 'A' / 'B' / 'True' / 'False' (case-insensitive)
  from the generation; compare to gold.
Blocksworld: extract action sequence (lines starting with '(' ); exact
  multiset match against ground truth (order doesn't matter for being a
  *valid* plan, but exact-prefix is easier to score and is what PlanBench
  uses for `task_1_plan_generation`).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def score_prontoqa(gen: str, gold: str) -> bool:
    g = gen.strip().splitlines()[0] if gen.strip() else ""
    # try direct letter
    m = re.search(r"\b([AB])\b", g)
    if m:
        return m.group(1) == gold
    # try true/false
    g_lower = g.lower()
    if "true" in g_lower and "false" not in g_lower:
        return gold == "A"
    if "false" in g_lower and "true" not in g_lower:
        return gold == "B"
    return False


_ACTION_RE = re.compile(r"^\([\w\- ]+\)\s*$")
# Natural-language patterns used by PlanBench in-context examples.
_NL_PICKUP = re.compile(r"pick(?:[ -])up the (\w+) block", re.I)
_NL_PUTDOWN = re.compile(r"put(?: down|down)? the (\w+) block", re.I)
_NL_STACK = re.compile(
    r"stack the (\w+) block on(?:to| top of)? the (\w+) block", re.I)
_NL_UNSTACK = re.compile(
    r"unstack the (\w+) block from (?:on top of |from )?the (\w+) block", re.I)


def _nl_line_to_pddl(ln: str) -> str | None:
    """Convert one natural-language Blocksworld action line to PDDL form.

    Returns None if no recognised action.
    """
    ln = ln.strip()
    # strip leading enumerators like "1. " "- " "* "
    ln = re.sub(r"^[\s\-\*\d\.]+", "", ln)
    if m := _NL_UNSTACK.search(ln):
        return f"(unstack {m.group(1).lower()} {m.group(2).lower()})"
    if m := _NL_STACK.search(ln):
        return f"(stack {m.group(1).lower()} {m.group(2).lower()})"
    if m := _NL_PICKUP.search(ln):
        return f"(pick-up {m.group(1).lower()})"
    if m := _NL_PUTDOWN.search(ln):
        return f"(put-down {m.group(1).lower()})"
    return None


def extract_blocksworld_plan(gen: str) -> list[str]:
    """Extract action sequence in canonical PDDL form.

    Accepts both the literal `(action args)` lines AND natural-language lines
    like 'unstack the blue block from on top of the orange block' (PlanBench's
    in-context example uses natural language, so the model copies that style).
    """
    out = []
    for ln in gen.split("\n"):
        ln_stripped = ln.strip()
        if not ln_stripped:
            continue
        if ln_stripped.startswith("[PLAN END]"):
            break
        if ln_stripped.startswith("[STATEMENT]") and out:
            break
        # Try PDDL format first
        if _ACTION_RE.match(ln_stripped):
            out.append(ln_stripped)
            continue
        # Try natural-language format
        pddl = _nl_line_to_pddl(ln_stripped)
        if pddl is not None:
            out.append(pddl)
    return out


def score_blocksworld(gen: str, gold: str) -> tuple[bool, dict]:
    pred = extract_blocksworld_plan(gen)
    gold_lines = [ln.strip() for ln in gold.split("\n") if ln.strip()]
    return pred == gold_lines, {
        "pred_len": len(pred), "gold_len": len(gold_lines),
        "pred": pred, "gold": gold_lines,
    }


def score_blocksworld_goal_reaching(gen: str, prompt: str) -> tuple[bool, dict]:
    """The proper Blocksworld metric: simulate the model's plan from the
    initial state, return True iff the goal is achieved (goal_facts ⊆
    final_state). Stops early on illegal action.

    This matches what PlanBench's full evaluator does (with Fast-Downward
    verification) — alternative valid plans count as correct, and the model
    is rewarded for reaching the goal even if it goes on to emit extra
    actions afterward (which is common; we accept goal-on-the-way plans).
    """
    from src.oracle_blocksworld import (
        parse_problem, apply_action, is_goal, Action,
    )
    pred = extract_blocksworld_plan(gen)
    try:
        problem = parse_problem(prompt)
    except Exception as e:
        return False, {"error": f"parse_problem failed: {e}",
                        "pred_len": len(pred)}
    state = problem.init
    goal_reached_at = -1
    for step_i, ln in enumerate(pred):
        inner = ln.strip("()").split()
        if not inner:
            break
        a = Action(op=inner[0], args=tuple(inner[1:]))
        new_state = apply_action(state, a)
        if new_state == state:
            # Illegal action (precondition not satisfied) — stop here.
            break
        state = new_state
        if is_goal(state, problem.goal):
            goal_reached_at = step_i + 1
            break
    return goal_reached_at >= 0, {
        "goal_reached_at": goal_reached_at, "pred_len": len(pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                     help="Path or glob to jsonl(s) with 'generation' + 'answer_label'")
    ap.add_argument("--task", required=True,
                     choices=["prontoqa", "blocksworld", "blocksworld_goal"])
    ap.add_argument("--show_failures", type=int, default=3)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.input))
    if not paths:
        paths = [Path(args.input)]
    records = []
    for p in paths:
        if not p.exists():
            continue
        records += [json.loads(l) for l in open(p)]
    if not records:
        print(f"No records found for {args.input}")
        return
    print(f"Scoring {len(records)} records ({args.task})")

    n_correct = 0
    failures = []
    for r in records:
        gen = r["generation"]
        gold = r["answer_label"]
        if args.task == "prontoqa":
            ok = score_prontoqa(gen, gold)
        elif args.task == "blocksworld":
            ok, _ = score_blocksworld(gen, gold)
        else:  # blocksworld_goal
            ok, _ = score_blocksworld_goal_reaching(gen, r["prompt"])
        if ok:
            n_correct += 1
        else:
            failures.append(r)

    print(f"  correct: {n_correct}/{len(records)} = {n_correct/len(records):.2%}")
    if failures and args.show_failures:
        print(f"\n  First {args.show_failures} failures:")
        for r in failures[: args.show_failures]:
            print(f"  --")
            print(f"  gold: {repr(r['answer_label'])[:100]}")
            print(f"  gen:  {repr(r['generation'])[:200]}")


if __name__ == "__main__":
    main()
