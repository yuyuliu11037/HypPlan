"""ProofWriter (Tafjord et al., 2021) oracle — Group B OOD #3.

We use the public AllenAI release at
[external/proofwriter/proofwriter-dataset-V2020.12.3/CWA/depth-3/](../external/proofwriter/proofwriter-dataset-V2020.12.3/),
specifically the **CWA depth-3** variant (committed in `docs/dataset_construction.md`).

Each ProofWriter "theory" record contains:
- A NL paragraph of facts and rules.
- Structured `triples` (initial facts) with `representation` `("S" "V" "O" "+/~")`.
- Structured `rules` with `representation`
  `(((premise1) (premise2 ...)) -> (conclusion))`.
- 16 questions per theory, each with a target representation, True/False
  answer, question depth (QDep), and a `proofsWithIntermediates` field
  giving the canonical derivation chain.

Per-question record schema (what the JSONL produced by
`data/import_proofwriter.py` emits, also what this oracle reads):
- `theory_text`: the NL paragraph.
- `initial_facts`: list of `(S, V, O, polarity)` tuples.
- `rule_texts`: dict `rule_id -> NL text` for rendering.
- `rules_struct`: dict `rule_id -> {"premises": [...], "conclusion": [...],
  "var": Optional[str]}` parsed from representation.
- `target`: `(S, V, O, polarity)` tuple — the question's target fact.
- `answer`: bool.
- `proof_chain`: ordered list of derivation steps for True-answer questions,
  each step `{"rule_id": str, "intermediate": (S, V, O, p), "intermediate_text": str}`.
  Empty for False or QDep=0 records.

The oracle exposes the standard Tree/Node interface:
- `enumerate_tree(problem)`: linear chain along the proof for True answers
  (length = QDep + 1 nodes), or a single-node tree for False / QDep=0
  questions. Stage-1 head training reads these caches.
- `winning_steps(state, problem)`: the next canonical proof step (always
  unique along the gold chain).

The model's task at eval time is only to output `Answer: True` or
`Answer: False` — we score the final answer letter, not the proof
itself, matching ProntoQA's scoring convention.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


Triple = tuple[str, str, str, str]   # (subject, verb, object, polarity '+'/'~')


# ---------------------------- Representation parsing ----------------------------

_TOKEN_RE = re.compile(r'"([^"]*)"')


def parse_triple(rep: str) -> Optional[Triple]:
    """Parse a single triple representation `("S" "V" "O" "+")` -> tuple."""
    toks = _TOKEN_RE.findall(rep)
    if len(toks) != 4:
        return None
    return (toks[0], toks[1], toks[2], toks[3])


def parse_rule(rep: str) -> Optional[dict]:
    """Parse a rule representation
    `(((p1) (p2)) -> (c))` into {"premises": [...], "conclusion": [...]}."""
    arrow = rep.find("->")
    if arrow < 0:
        return None
    head = rep[:arrow]
    body = rep[arrow + 2:].strip().strip("()")
    conclusion = parse_triple("(" + body + ")")
    if conclusion is None:
        return None
    # Premises are inner triples within the head.
    pre_rep = head.strip().strip("()")
    # Each premise is wrapped in parens; find them by scanning.
    depth = 0
    start = -1
    raw_pres: list[str] = []
    for i, ch in enumerate(pre_rep):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                raw_pres.append(pre_rep[start:i + 1])
                start = -1
    premises: list[Triple] = []
    for pr in raw_pres:
        t = parse_triple(pr)
        if t is None:
            return None
        premises.append(t)
    return {"premises": tuple(premises), "conclusion": conclusion}


# ---------------------------- Data classes ----------------------------

@dataclass
class Problem:
    theory_text: str
    initial_facts: tuple[Triple, ...]
    rule_texts: dict[str, str]                   # rule_id -> NL
    rules_struct: dict[str, dict]                # rule_id -> {"premises", "conclusion"}
    triple_texts: dict[Triple, str]              # triple -> NL
    target: Triple
    target_text: str                              # NL of question
    answer: bool
    proof_chain: tuple[dict, ...]                # ordered derivations
    raw: str = ""

    def render_problem(self) -> str:
        return (
            f"{self.theory_text}\n\n"
            f"Question: {self.target_text}"
            f"\nIs the statement above true or false?"
        )


@dataclass
class Node:
    node_id: int
    state: frozenset[Triple]
    parent: Optional[int]
    rule_id: Optional[str]
    intermediate: Optional[Triple]
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_solved: bool = False
    v_value: int = -1


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


# ---------------------------- Predicates ----------------------------

def is_solved(state: frozenset[Triple], target: Triple, answer: bool,
              proof_chain: tuple = ()) -> bool:
    """For True answers with a non-empty proof: target ∈ state.

    For True answers with an empty proof_chain (e.g., negative-polarity
    targets True under closed-world negation-as-failure, or QDep=0
    direct facts), trivially solved at the initial state.

    For False answers: trivially solved at the initial state (we don't
    enumerate the closure here)."""
    if not answer:
        return True
    if not proof_chain:
        return True
    return target in state


# ---------------------------- Tree enumeration ----------------------------

def enumerate_tree(problem: Problem, max_nodes: int = 50) -> Tree:
    """Build a linear chain along the gold proof. For False / QDep=0 records
    the tree is a single root node."""
    root_state = frozenset(problem.initial_facts)
    root = Node(
        node_id=0,
        state=root_state,
        parent=None,
        rule_id=None,
        intermediate=None,
        depth=0,
        is_solved=is_solved(root_state, problem.target, problem.answer, problem.proof_chain),
    )
    nodes: list[Node] = [root]

    if problem.answer and problem.proof_chain:
        cur_state = root_state
        cur_id = 0
        for d, step in enumerate(problem.proof_chain):
            new_state = cur_state | {step["intermediate"]}
            nid = len(nodes)
            nodes.append(
                Node(
                    node_id=nid,
                    state=new_state,
                    parent=cur_id,
                    rule_id=step["rule_id"],
                    intermediate=step["intermediate"],
                    depth=d + 1,
                    is_solved=is_solved(new_state, problem.target, True,
                                          problem.proof_chain),
                )
            )
            nodes[cur_id].children.append(nid)
            cur_state = new_state
            cur_id = nid
            if len(nodes) >= max_nodes:
                break

    dist = [-1] * len(nodes)
    bq: deque[int] = deque()
    for n in nodes:
        if n.is_solved:
            dist[n.node_id] = 0
            bq.append(n.node_id)
    while bq:
        nid = bq.popleft()
        n = nodes[nid]
        nbrs = list(n.children)
        if n.parent is not None:
            nbrs.append(n.parent)
        for m in nbrs:
            if dist[m] == -1:
                dist[m] = dist[nid] + 1
                bq.append(m)
    for n in nodes:
        n.v_value = dist[n.node_id]
    return Tree(problem=problem, nodes=nodes)


# ---------------------------- Lightweight oracle ----------------------------

def _state_after_step_count(problem: Problem, k: int) -> frozenset[Triple]:
    """State after applying the first k proof steps (k=0 = initial)."""
    state = frozenset(problem.initial_facts)
    for i in range(min(k, len(problem.proof_chain))):
        state = state | {problem.proof_chain[i]["intermediate"]}
    return state


def winning_steps(state: frozenset[Triple], problem: Problem) -> list[dict]:
    """Return the NEXT step in the gold proof chain (unique along the path).
    For False or QDep=0 records, returns []."""
    if not problem.answer or not problem.proof_chain:
        return []
    if is_solved(state, problem.target, problem.answer, problem.proof_chain):
        return []
    derived = state - frozenset(problem.initial_facts)
    next_idx = len(derived)
    if next_idx >= len(problem.proof_chain):
        return []
    return [problem.proof_chain[next_idx]]


def validate_step(state: frozenset[Triple], step: dict,
                   problem: Problem) -> tuple[bool, frozenset[Triple]]:
    """Verify the step matches the next gold derivation."""
    wins = winning_steps(state, problem)
    if not wins:
        return False, state
    if wins[0]["rule_id"] != step["rule_id"]:
        return False, state
    if wins[0]["intermediate"] != step["intermediate"]:
        return False, state
    return True, state | {step["intermediate"]}


# ---------------------------- Rendering / parsing ----------------------------

def render_state(problem: Problem, state: frozenset[Triple]) -> str:
    derived = state - frozenset(problem.initial_facts)
    derived_texts = []
    for t in derived:
        # Look up NL text from problem.triple_texts (proof intermediates) if
        # available, else render the raw representation.
        if t in problem.triple_texts:
            derived_texts.append(problem.triple_texts[t])
        else:
            derived_texts.append(_render_triple(t))
    return (
        f"Question: {problem.target_text}\n"
        f"Derived so far: "
        f"{'; '.join(derived_texts) if derived_texts else '(none)'}"
    )


def _render_triple(t: Triple) -> str:
    s, v, o, p = t
    neg = "" if p == "+" else "not "
    return f"{s} {neg}{v} {o}"


def format_question(problem: Problem) -> str:
    return problem.render_problem()


def format_step_text(step: dict) -> str:
    """Render one proof step in NL: 'apply rule_id: derive <intermediate>'."""
    rid = step["rule_id"]
    text = step.get("intermediate_text") or _render_triple(step["intermediate"])
    return f"apply {rid}: {text}"


def format_gold_trajectory(problem: Problem) -> str:
    lines: list[str] = []
    if problem.answer and problem.proof_chain:
        for i, step in enumerate(problem.proof_chain):
            lines.append(f"Step {i + 1}: {format_step_text(step)}")
    ans = "True" if problem.answer else "False"
    lines.append(f"Answer: {ans}")
    return "\n".join(lines)


_ANSWER_RE = re.compile(r"Answer\s*[:\-]?\s*(True|False)\b", re.IGNORECASE)


def parse_answer(generation: str) -> Optional[bool]:
    m = _ANSWER_RE.search(generation)
    if m is None:
        return None
    return m.group(1).lower() == "true"


def score_answer(prediction: Optional[bool], gold: bool) -> bool:
    return prediction is not None and prediction == gold


# ---------------------------- Step parsing ----------------------------

_STEP_RULE_RE = re.compile(r"apply\s+(rule\d+)", re.IGNORECASE)


def parse_step(text: str, problem: Problem,
                state: frozenset[Triple]) -> Optional[dict]:
    """Parse one step body into a derivation dict. Lenient: accepts
    'apply ruleN' format and matches against the canonical gold step."""
    wins = winning_steps(state, problem)
    if not wins:
        return None
    expected = wins[0]
    m = _STEP_RULE_RE.search(text)
    if m is None:
        return None
    if m.group(1).lower() != expected["rule_id"].lower():
        return None
    return expected
