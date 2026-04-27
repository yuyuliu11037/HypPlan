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


def score_rulechain(gen: str, record: dict) -> tuple[bool, dict]:
    """Rule-chaining / synthlogic: succeed iff the generation derives the
    target predicate via legal forward-chaining steps from the initial facts.

    Reconstructs the rule book from the record, walks the model's emitted
    steps, and checks whether `target` ends up in state. Tolerates extra
    steps after the target is derived (early-stop on success)."""
    from src.oracle_rulechain import (
        Problem, Rule, applicable_rules, apply_rule, decidable, parse_step,
    )
    rules = tuple(
        Rule(premises=tuple(sorted(r["premises"])), conclusion=r["conclusion"])
        for r in record["rules"]
    )
    problem = Problem(
        initial_facts=frozenset(record["initial_facts"]),
        target=record["target"],
        rules=rules,
    )
    state = problem.initial_facts
    n_legal = 0
    for line in gen.splitlines():
        if decidable(state, problem.target):
            break
        step = parse_step(line, problem)
        if step is None:
            continue
        # legality: premises in state, conclusion new, and step is in rules
        if step not in problem.rules:
            continue
        if step.conclusion in state:
            continue
        if not all(p in state for p in step.premises):
            continue
        state = apply_rule(state, step)
        n_legal += 1
    ok = decidable(state, problem.target)
    return ok, {"n_legal_steps": n_legal, "final_state_size": len(state)}


def score_numpath(gen: str, record: dict) -> tuple[bool, dict]:
    """Number-path: simulate the model's emitted ops from `start` and check
    whether the final value equals `target`. Each op must be in the
    allowed set."""
    from src.oracle_numpath import Op, Problem
    ops = tuple(Op(o["kind"], int(o["const"])) for o in record["ops"])
    problem = Problem(start=int(record["start"]),
                      target=int(record["target"]),
                      ops=ops,
                      max_value=int(record.get("max_value", 999)))
    state = problem.start
    step_re = re.compile(
        r"(\d+)\s*([\+\-\*xX/])\s*(\d+)\s*=\s*(-?\d+)", re.IGNORECASE
    )
    last_value = state
    for line in gen.splitlines():
        m = step_re.search(line)
        if m is None:
            continue
        a = int(m.group(1)); sym = m.group(2).lower(); b = int(m.group(3))
        r = int(m.group(4))
        if a != state:
            break
        kind = {"+": "ADD", "-": "SUB", "*": "MUL", "x": "MUL",
                "/": "DIV"}.get(sym)
        if kind is None:
            break
        op = Op(kind=kind, const=b)
        if op not in problem.ops:
            break
        ns = op.apply(state, problem.max_value)
        if ns is None or ns != r:
            break
        state = ns
        last_value = state
        if state == problem.target:
            break
    ok = (state == problem.target)
    return ok, {"final": state, "target": problem.target}
    """Small-scale Countdown: simulate the model's emitted steps from `pool`
    and check whether the final single number equals `target`.

    Lenient parsing: matches lines of the form "a op b = r" where op is
    +, -, *, /, x, X. Uses each operand at most once per step. Stops on the
    first illegal step or when only one number remains."""
    pool = list(record["pool"])
    target = int(record["target"])

    step_re = re.compile(
        r"(\d+)\s*([\+\-\*xX/])\s*(\d+)\s*=\s*(-?\d+)", re.IGNORECASE
    )
    state = pool[:]
    last_value = None
    for line in gen.splitlines():
        m = step_re.search(line)
        if m is None:
            continue
        a = int(m.group(1)); op = m.group(2).lower(); b = int(m.group(3))
        r = int(m.group(4))
        if op in ("x",): op = "*"
        # legality: both operands must be in current state
        try:
            ai = state.index(a)
            tmp = list(state); tmp.pop(ai)
            bi = tmp.index(b)
        except ValueError:
            break
        if op == "+":
            r_chk = a + b
        elif op == "-":
            r_chk = a - b
        elif op == "*":
            r_chk = a * b
        elif op == "/":
            if b == 0 or a % b != 0:
                break
            r_chk = a // b
        else:
            break
        if r_chk != r:
            break
        # Apply: remove a and b, add r
        state.remove(a); state.remove(b); state.append(r)
        last_value = r

    final_value = state[0] if len(state) == 1 else last_value
    ok = final_value is not None and final_value == target
    return ok, {"final": final_value, "target": target,
                 "remaining_pool": state}


def score_g24(gen: str, record: dict) -> tuple[bool, dict]:
    """Game-of-24 / cd_small-style: simulate model's emitted ops on the
    pool, succeed iff the final single number equals 24.

    Two schemas supported:
      `data/24_test.jsonl`: {"problem": "a,b,c,d", "text": ..., ...} — target=24.
      `data/24_varied_bal_test.jsonl`: {"pool": [...], "target": int, ...}.
    """
    if "pool" in record:
        return score_cd_small(gen, record)
    # 24_test.jsonl schema: parse "problem" string into pool, target=24.
    pool = [int(x) for x in record["problem"].split(",")]
    rec2 = dict(record)
    rec2["pool"] = pool
    rec2["target"] = int(record.get("target", 24))
    return score_cd_small(gen, rec2)


def score_proofwriter(gen: str, record: dict) -> tuple[bool, dict]:
    """ProofWriter (CWA): extract `Answer: True/False` and compare to gold."""
    from src.oracle_proofwriter import parse_answer, score_answer
    pred = parse_answer(gen)
    gold = bool(record["answer"])
    ok = score_answer(pred, gold)
    return ok, {"pred": pred, "gold": gold}


def score_lineq(gen: str, record: dict) -> tuple[bool, dict]:
    """Linear equations: extract `Answer: x = K` and compare to gold int."""
    from src.oracle_lineq import parse_answer, score_answer
    pred = parse_answer(gen)
    gold = int(record["solution"])
    ok = score_answer(pred, gold)
    return ok, {"pred": pred, "gold": gold}


def score_clutrr(gen: str, record: dict) -> tuple[bool, dict]:
    """CLUTRR-like: parse the predicted relation from the generation and
    compare to the gold answer. Lenient: accepts the relation appearing
    anywhere in the answer line."""
    from src.oracle_clutrr import parse_answer, score_answer
    pred = parse_answer(gen)
    gold = record["answer"]
    ok = score_answer(pred, gold)
    return ok, {"pred": pred, "gold": gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                     help="Path or glob to jsonl(s) with 'generation' + 'answer_label'")
    ap.add_argument("--task", required=True,
                     choices=["prontoqa", "blocksworld", "blocksworld_goal",
                              "graphcolor", "rulechain", "synthlogic",
                              "clutrr", "lineq", "proofwriter"])
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
        elif args.task == "blocksworld_goal":
            ok, _ = score_blocksworld_goal_reaching(gen, r["prompt"])
        elif args.task == "graphcolor":
            from src.oracle_graphcolor import (
                Problem, parse_coloring, score_coloring,
            )
            p = Problem(n=r["n"],
                         edges=tuple(map(tuple, r["edges"])))
            coloring = parse_coloring(gen, p)
            ok = score_coloring(p, coloring)
        elif args.task in ("rulechain", "synthlogic"):
            ok, _ = score_rulechain(gen, r)
        elif args.task == "clutrr":
            ok, _ = score_clutrr(gen, r)
        elif args.task == "lineq":
            ok, _ = score_lineq(gen, r)
        else:  # proofwriter
            ok, _ = score_proofwriter(gen, r)
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
