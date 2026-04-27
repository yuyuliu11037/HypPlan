"""Build planning-token SFT training data for the 3 OOD tasks.

For each task, output a jsonl with schema {"question": str, "answer": str},
where the `answer` interleaves planning tokens with the step-by-step solution.

Tasks:
- Countdown: reuse data/cd_train_sft.jsonl. Planning tokens = arithmetic
  operator before each `Step N:` line, plus <PLAN:ANS> before "Answer:".
- ProntoQA: generate forward-chaining proofs from the oracle. Planning
  tokens = <PLAN:DERIVE_TRUE> / <PLAN:DERIVE_FALSE> based on conclusion value.
- Blocksworld: use PlanBench gold plans (records 200-449, same train slice
  as head training). Planning tokens = action type
  <PLAN:PICKUP/PUTDOWN/STACK/UNSTACK>, plus <PLAN:ANS> at end.

Outputs:
  data/cd_train_sft_plan.jsonl
  data/prontoqa_train_sft_plan.jsonl
  data/blocksworld_train_sft_plan.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------- Countdown ----------

_CD_STEP_RE = re.compile(
    r"Step\s+\d+:\s+\S+\s+([+\-*/])\s+\S+\s+=\s+\S+")


def insert_pt_cd(text: str) -> str:
    """Prefix each 'Step N: a op b = r' with <PLAN:op>; <PLAN:ANS> before Answer."""
    out_parts = []
    last_end = 0
    for m in _CD_STEP_RE.finditer(text):
        out_parts.append(text[last_end : m.start()])
        out_parts.append(f"<PLAN:{m.group(1)}> ")
        out_parts.append(m.group(0))
        last_end = m.end()
    out_parts.append(text[last_end:])
    out = "".join(out_parts)
    out = out.replace("Answer:", "<PLAN:ANS> Answer:", 1)
    return out


def make_cd_question(text: str) -> str:
    """The CD trajectory text starts with 'Problem: ...'. Strip that off as
    the question; the rest is the answer."""
    if text.startswith("Problem: "):
        # Find first \n and split
        i = text.find("\n")
        return text[len("Problem: "): i].strip(), text[i + 1:]
    return "", text


def build_cd():
    out_path = Path("data/cd_train_sft_plan.jsonl")
    n = 0
    with open("data/cd_train_sft.jsonl") as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            q, a = make_cd_question(rec["text"])
            if not q:
                continue
            a_plan = insert_pt_cd(a)
            fout.write(json.dumps({
                "question": q, "answer": a_plan,
                "pool": rec["pool"], "target": rec["target"],
            }) + "\n")
            n += 1
    print(f"CD: wrote {n} records → {out_path}")
    print(f"  Sample answer:\n  {json.loads(open(out_path).readline())['answer'][:300]}")


# ---------- Blocksworld ----------

_BW_ACTION_TAG = {
    "pick-up": "PICKUP",
    "put-down": "PUTDOWN",
    "stack": "STACK",
    "unstack": "UNSTACK",
}


def render_bw_action_nl(action_tuple: tuple[str, ...]) -> str:
    """Render PDDL action `(op args...)` as natural-language sentence."""
    op, *args = action_tuple
    if op == "pick-up":
        return f"pick up the {args[0]} block"
    if op == "put-down":
        return f"put down the {args[0]} block"
    if op == "stack":
        return f"stack the {args[0]} block on top of the {args[1]} block"
    if op == "unstack":
        return f"unstack the {args[0]} block from on top of the {args[1]} block"
    return " ".join(action_tuple)


def parse_pddl_line(ln: str) -> tuple[str, ...] | None:
    ln = ln.strip()
    if not (ln.startswith("(") and ln.endswith(")")):
        return None
    inner = ln[1:-1].split()
    return tuple(inner)


def build_bw():
    from datasets import load_dataset
    bw = load_dataset("tasksource/planbench", "task_1_plan_generation",
                       split="train")
    bw_recs = [ex for ex in bw if ex["domain"] == "blocksworld"
               and ex["prompt_type"] == "oneshot"]
    rng = random.Random(1234)
    rng.shuffle(bw_recs)
    train = bw_recs[200:450]   # same slice as head training
    out_path = Path("data/blocksworld_train_sft_plan.jsonl")
    n = 0
    with out_path.open("w") as fout:
        for rec in train:
            # Question = the full PlanBench query (model sees it at eval)
            q = rec["query"]
            # Answer = NL plan + planning tokens
            actions = []
            for ln in rec["ground_truth_plan"].split("\n"):
                a = parse_pddl_line(ln)
                if a is None:
                    continue
                op = a[0]
                tag = _BW_ACTION_TAG.get(op, "OTHER")
                actions.append(f"<PLAN:{tag}> {render_bw_action_nl(a)}")
            a_text = "\n".join(actions) + "\n<PLAN:ANS> [PLAN END]"
            fout.write(json.dumps({
                "question": q, "answer": a_text,
                "ground_truth_plan": rec["ground_truth_plan"],
            }) + "\n")
            n += 1
    print(f"BW: wrote {n} records → {out_path}")
    print(f"  Sample answer:\n  {json.loads(open(out_path).readline())['answer'][:400]}")


# ---------- ProntoQA ----------

def _format_predicate_phrase(pred: str, val: bool) -> str:
    verb = "is" if val else "is not"
    return f"{verb} {pred.lower()}"


def render_pq_proof(problem) -> tuple[str, str]:
    """Greedy forward-chaining proof. Picks rules in encounter order until
    the query is decidable. Returns (question, answer_with_planning_tokens).
    """
    from src.oracle_pronto import forward_apply, decidable
    state = problem.facts
    facts_text = ", ".join(
        f"{problem.entity} {_format_predicate_phrase(p, v)}"
        for (p, v) in sorted(state)
    )
    qpred, qval = problem.query
    question = (
        f"Initial facts: {facts_text}.\n"
        f"Question: is {problem.entity} {_format_predicate_phrase(qpred, qval)}?"
    )

    steps = []
    safety = 30
    while safety > 0 and not decidable(state, problem.query):
        applied = False
        for rule in problem.rules:
            new = forward_apply(state, rule)
            if new is None or new == state:
                continue
            tag = "DERIVE_TRUE" if rule.conclusion_val else "DERIVE_FALSE"
            steps.append(
                f"<PLAN:{tag}> Step {len(steps) + 1}: "
                f"since {problem.entity} "
                f"{_format_predicate_phrase(rule.premise_pred, rule.premise_val)}, "
                f"{problem.entity} "
                f"{_format_predicate_phrase(rule.conclusion_pred, rule.conclusion_val)}."
            )
            state = new
            applied = True
            break
        if not applied:
            break
        safety -= 1
    # Final answer
    final_pred_in_state = next(((p, v) for (p, v) in state if p == qpred), None)
    if final_pred_in_state is None:
        # Unprovable — answer "B" (False) by default
        answer_letter = "B"
    else:
        answer_letter = "A" if final_pred_in_state[1] == qval else "B"
    steps.append(f"<PLAN:ANS> Answer: {answer_letter}")
    return question, "\n".join(steps)


def build_pq():
    from datasets import load_dataset
    from src.oracle_pronto import parse_problem
    ds = load_dataset("renma/ProntoQA", split="validation")
    rng = random.Random(1234)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    train = [ds[i] for i in indices[:250]]   # same slice as head training
    out_path = Path("data/prontoqa_train_sft_plan.jsonl")
    n = 0
    with out_path.open("w") as fout:
        for rec in train:
            try:
                p = parse_problem(rec["raw_logic_programs"])
                q, a = render_pq_proof(p)
            except Exception as e:
                continue
            fout.write(json.dumps({
                "question": q, "answer": a, "id": rec["id"],
                "answer_letter": rec["answer"],
            }) + "\n")
            n += 1
    print(f"PQ: wrote {n} records → {out_path}")
    print(f"  Sample answer:\n  {json.loads(open(out_path).readline())['answer'][:400]}")


if __name__ == "__main__":
    build_cd()
    build_bw()
    build_pq()
