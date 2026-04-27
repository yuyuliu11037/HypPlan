"""Oracle for synthetic rule-chaining (Horn-clause forward chaining).

Group B training-source analog of [src/oracle_24_varied.py](src/oracle_24_varied.py),
plus the heavy Tree/Node interface that mirrors [src/oracle_pronto.py](src/oracle_pronto.py).

A problem is `(initial_facts, target, rules)`:
- `initial_facts`: frozenset of base predicates known at the start.
- `target`: predicate we want to derive.
- `rules`: tuple of Horn clauses, each of the form
  "if A (and B) then C" with 1 or 2 premises and 1 conclusion.

A state is a frozenset of currently-known predicates. Forward chaining adds
the conclusion of any rule whose premises are all in state. Goal: `target` is
in state. The state-space is monotone (only grows), so reachability has a
unique min-depth.

This oracle exposes both:
- The Tree/Node interface (`enumerate_tree`) that Stage-1 head training, the
  in-domain eval driver, and the per-task adapter consume.
- A lightweight `winning_steps(state, problem)` lookup used by the DAgger
  rollout loop during Stage-2 varied training (analog of `oracle_24_varied`).

Predicate names are abstract symbols (`p0`, `p1`, ...). The varied-target
training data fixes the rule book per-problem and varies the target; this is
the "varied" trick that prevented G24 memorization in Group A.
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


Predicate = str


# ---------------------------- Data classes ----------------------------

@dataclass(frozen=True)
class Rule:
    premises: tuple[Predicate, ...]  # 1 or 2 premises (Horn)
    conclusion: Predicate

    def render(self) -> str:
        if len(self.premises) == 1:
            return f"if {self.premises[0]}, then {self.conclusion}"
        return f"if {self.premises[0]} and {self.premises[1]}, then {self.conclusion}"


@dataclass
class Problem:
    initial_facts: frozenset[Predicate]
    target: Predicate
    rules: tuple[Rule, ...]
    raw: str = ""

    def render_problem(self) -> str:
        lines = ["Rules:"]
        for r in self.rules:
            lines.append(f"- {r.render()}")
        lines.append(f"Initial facts: {', '.join(sorted(self.initial_facts))}")
        lines.append(f"Goal: derive {self.target}")
        return "\n".join(lines)


@dataclass
class Node:
    node_id: int
    state: frozenset[Predicate]
    parent: Optional[int]
    rule_used: Optional[Rule]
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_solved: bool = False
    v_value: int = -1


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


# ---------------------------- Forward chaining ----------------------------

def applicable_rules(
    state: frozenset[Predicate], rules: tuple[Rule, ...]
) -> list[Rule]:
    """Rules whose premises are all in state and whose conclusion is new.

    Deduplicates rules with the same conclusion: only the first such rule
    is returned (state is a set, multiple rules concluding the same
    predicate produce the same successor)."""
    out: list[Rule] = []
    seen_concs: set[Predicate] = set()
    for r in rules:
        if r.conclusion in state or r.conclusion in seen_concs:
            continue
        if all(p in state for p in r.premises):
            seen_concs.add(r.conclusion)
            out.append(r)
    return out


def apply_rule(state: frozenset[Predicate], rule: Rule) -> frozenset[Predicate]:
    return state | {rule.conclusion}


def decidable(state: frozenset[Predicate], target: Predicate) -> bool:
    """Solution check (named `decidable` to match `oracle_pronto`)."""
    return target in state


def render_state(problem: Problem, state: frozenset[Predicate]) -> str:
    derived = sorted(state - problem.initial_facts)
    initial = sorted(problem.initial_facts)
    return (
        f"Initial facts: {', '.join(initial)}\n"
        f"Derived so far: {', '.join(derived) if derived else '(none)'}\n"
        f"Goal: derive {problem.target}"
    )


# ---------------------------- Tree enumeration ----------------------------

def enumerate_tree(
    problem: Problem, max_nodes: int = 5000, max_depth: int = 8
) -> Tree:
    """BFS-enumerate the reachable state-space and label v_value as the BFS
    distance (over undirected parent<->child edges) to the nearest solved
    node. Mirrors the labeling convention of `oracle_pronto.enumerate_tree`."""
    root_state = frozenset(problem.initial_facts)
    nodes: list[Node] = [
        Node(
            node_id=0,
            state=root_state,
            parent=None,
            rule_used=None,
            depth=0,
            is_solved=decidable(root_state, problem.target),
        )
    ]
    seen: dict[frozenset, int] = {root_state: 0}
    queue: deque[int] = deque([0])

    while queue and len(nodes) < max_nodes:
        nid = queue.popleft()
        node = nodes[nid]
        if node.is_solved or node.depth >= max_depth:
            continue
        for rule in applicable_rules(node.state, problem.rules):
            new_state = apply_rule(node.state, rule)
            if new_state in seen:
                child_id = seen[new_state]
            else:
                child_id = len(nodes)
                nodes.append(
                    Node(
                        node_id=child_id,
                        state=new_state,
                        parent=nid,
                        rule_used=rule,
                        depth=node.depth + 1,
                        is_solved=decidable(new_state, problem.target),
                    )
                )
                seen[new_state] = child_id
                queue.append(child_id)
                if len(nodes) >= max_nodes:
                    break
            if child_id not in node.children:
                node.children.append(child_id)

    # Backward BFS over undirected parent/child edges to assign v_value.
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

def _min_dist_to_target(
    start: frozenset[Predicate],
    problem: Problem,
    max_nodes: int = 5000,
) -> Optional[int]:
    """Forward BFS distance from `start` to any state containing target.
    Returns None if target unreachable within the search budget."""
    if decidable(start, problem.target):
        return 0
    seen: dict[frozenset, int] = {start: 0}
    q: deque[frozenset] = deque([start])
    while q and len(seen) < max_nodes:
        s = q.popleft()
        d = seen[s]
        for r in applicable_rules(s, problem.rules):
            ns = apply_rule(s, r)
            if ns in seen:
                continue
            nd = d + 1
            if decidable(ns, problem.target):
                return nd
            seen[ns] = nd
            q.append(ns)
    return None


def winning_steps(
    state: frozenset[Predicate], problem: Problem, max_nodes: int = 5000
) -> list[Rule]:
    """Rules applicable to `state` whose application strictly decreases the
    BFS distance to a target-reaching state (i.e., are on a shortest path).

    Used by the DAgger rollout adapter to label gold next-steps during
    Stage-2 varied training."""
    if decidable(state, problem.target):
        return []
    state_d = _min_dist_to_target(state, problem, max_nodes)
    if state_d is None:
        return []
    out: list[Rule] = []
    for r in applicable_rules(state, problem.rules):
        ns = apply_rule(state, r)
        ns_d = _min_dist_to_target(ns, problem, max_nodes)
        if ns_d is not None and ns_d == state_d - 1:
            out.append(r)
    return out


def validate_step(
    state: frozenset[Predicate],
    rule: Rule,
    problem: Problem,
) -> tuple[bool, frozenset[Predicate]]:
    """Verify `rule` is applicable in `state` (premises satisfied, conclusion
    new). Returns (legal, new_state). On illegal step, returns (False, state)."""
    if rule not in problem.rules:
        return False, state
    if rule.conclusion in state:
        return False, state
    if not all(p in state for p in rule.premises):
        return False, state
    return True, apply_rule(state, rule)


# ---------------------------- Problem generator ----------------------------

def generate_problem(
    n_predicates: int = 16,
    n_rules: int = 18,
    n_initial_facts: int = 4,
    target_depth: int = 2,
    seed: int = 0,
    max_attempts: int = 400,
    pred_prefix: str = "p",
) -> Problem:
    """Generate a rule-chaining problem whose target is reachable in
    EXACTLY `target_depth` forward-chaining steps from the initial facts.

    The shaping mirrors `24_varied_bal`: same overall structure, varied
    target, balanced depth distribution at the dataset level.

    `pred_prefix` controls the surface-form predicate names so eval-time
    problems can use a different vocabulary (e.g., "q") from training-time
    problems ("p") — this prevents the LoRA from generalizing via
    token-level predicate familiarity rather than reasoning structure."""
    rng = random.Random(seed)
    preds = [f"{pred_prefix}{i}" for i in range(n_predicates)]

    for _ in range(max_attempts):
        rules: list[Rule] = []
        seen_keys: set = set()
        # Sample a rule book.
        while len(rules) < n_rules:
            n_premises = 2 if rng.random() < 0.7 else 1
            premises = tuple(sorted(rng.sample(preds, n_premises)))
            conclusion = rng.choice(preds)
            if conclusion in premises:
                continue
            key = (premises, conclusion)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rules.append(Rule(premises, conclusion))

        rules_t = tuple(rules)
        initial = frozenset(rng.sample(preds, n_initial_facts))

        # BFS layered by depth: layers[d] = states first reached at depth d.
        layers: list[set[frozenset]] = [{initial}]
        seen_layer = {initial: 0}
        for d in range(target_depth):
            cur: set[frozenset] = set()
            for s in layers[-1]:
                for r in applicable_rules(s, rules_t):
                    ns = apply_rule(s, r)
                    if ns not in seen_layer:
                        seen_layer[ns] = d + 1
                        cur.add(ns)
            layers.append(cur)
            if not cur:
                break

        if len(layers) - 1 < target_depth or not layers[target_depth]:
            continue

        # Predicates first derivable at exactly `target_depth`.
        derivable_at_or_before: set[Predicate] = set()
        for d in range(target_depth):
            for s in layers[d]:
                derivable_at_or_before |= s
        candidates: set[Predicate] = set()
        for s in layers[target_depth]:
            candidates |= s - derivable_at_or_before
        if not candidates:
            continue
        target = rng.choice(sorted(candidates))

        return Problem(initial_facts=initial, target=target, rules=rules_t)

    raise RuntimeError(
        f"Could not generate rule-chaining problem at depth={target_depth} "
        f"after {max_attempts} attempts"
    )


# ---------------------------- Step parsing (NL <-> Rule) ----------------------------

_STEP_DERIVE_RE = re.compile(
    r"derive\s+([a-zA-Z]+\d+)\s+(?:from|using)\s+([^.\n]+)", re.IGNORECASE
)
_STEP_APPLY_RE = re.compile(
    r"apply\s+rule\s*[:\-]?\s*if\s+([^,]+?)(?:\s+and\s+([^,]+?))?\s*,?\s*then\s+([a-zA-Z]+\d+)",
    re.IGNORECASE,
)
_PREDICATE_TOKEN_RE = re.compile(r"[a-zA-Z]+\d+")


def format_step_text(rule: Rule) -> str:
    """Canonical NL rendering of a forward-chaining step."""
    if len(rule.premises) == 1:
        return f"apply rule: if {rule.premises[0]}, then {rule.conclusion}"
    return (
        f"apply rule: if {rule.premises[0]} and {rule.premises[1]}, "
        f"then {rule.conclusion}"
    )


def parse_step(text: str, problem: Problem) -> Optional[Rule]:
    """Parse one step of model output into a Rule from `problem.rules`.

    Matches both
      "apply rule: if A and B, then C"
      "derive C from A and B"
    """
    m = _STEP_APPLY_RE.search(text)
    if m:
        a, b, c = m.group(1).strip(), m.group(2), m.group(3).strip()
        prems = (a, b.strip()) if b else (a,)
        prems = tuple(sorted(prems))
        for r in problem.rules:
            if r.conclusion == c and tuple(sorted(r.premises)) == prems:
                return r
        return None
    m = _STEP_DERIVE_RE.search(text)
    if m:
        c = m.group(1).strip()
        prem_text = m.group(2).strip()
        toks = _PREDICATE_TOKEN_RE.findall(prem_text)
        prems = tuple(sorted(toks))
        for r in problem.rules:
            if r.conclusion == c and tuple(sorted(r.premises)) == prems:
                return r
    return None
