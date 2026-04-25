"""ProntoQA oracle: parse raw_logic_programs, run forward-chaining derivation.

A ProntoQA problem is a Horn-clause-like deductive setup:
  Facts: P(entity, True)        # one starting fact
  Rules: P($x, V) >>> Q($x, V') # Horn clauses about a single variable
  Query: Q(entity, V)            # what we want to verify

Since the entity is fixed across the whole problem, we represent state as a
frozenset of (predicate, bool) tuples. The tree branches over which rule we
apply next.

Public API:
- parse_problem(raw_logic_programs: str | list[str]) -> Problem
- forward_apply(state, rule) -> new state (or None if rule's premise unsatisfied
  or its conclusion already in state)
- decidable(state, query) -> bool — does the state contain query or its negation?
- enumerate_tree(problem) -> Tree (list of nodes with parent / children / depth /
  v_value = BFS distance to nearest decidable state)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


_FACT_RE = re.compile(r"(\w+)\(([\w\$]+),\s*(True|False)\)")
_RULE_RE = re.compile(
    r"(\w+)\(\$x,\s*(True|False)\)\s*>>>\s*(\w+)\(\$x,\s*(True|False)\)")


@dataclass(frozen=True)
class Rule:
    premise_pred: str
    premise_val: bool
    conclusion_pred: str
    conclusion_val: bool

    def render(self) -> str:
        prem = "is" if self.premise_val else "is not"
        conc = "is" if self.conclusion_val else "is not"
        return (f"if x {prem} a {self.premise_pred.lower()}, "
                f"then x {conc} {self.conclusion_pred.lower()}")


@dataclass
class Problem:
    entity: str
    facts: frozenset[tuple[str, bool]]   # initial known (predicate, value) pairs
    rules: tuple[Rule, ...]
    query: tuple[str, bool]              # (predicate, value) we want to verify
    raw: str = ""


def parse_problem(raw_logic_programs) -> Problem:
    if isinstance(raw_logic_programs, list):
        raw = raw_logic_programs[0]
    else:
        raw = raw_logic_programs
    sections = {}
    cur = None
    for ln in raw.split("\n"):
        if ln.endswith(":") and not ln.startswith(" "):
            cur = ln[:-1].strip()
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(ln)

    facts_lines = sections.get("Facts", [])
    rules_lines = sections.get("Rules", [])
    query_lines = sections.get("Query", [])

    # Parse single fact, capture entity.
    entity = None
    facts: set[tuple[str, bool]] = set()
    for ln in facts_lines:
        m = _FACT_RE.search(ln)
        if m:
            pred, ent, val = m.group(1), m.group(2), m.group(3) == "True"
            entity = ent
            facts.add((pred, val))

    rules: list[Rule] = []
    for ln in rules_lines:
        m = _RULE_RE.search(ln)
        if m:
            rules.append(Rule(
                premise_pred=m.group(1),
                premise_val=(m.group(2) == "True"),
                conclusion_pred=m.group(3),
                conclusion_val=(m.group(4) == "True"),
            ))

    qpred, qval = None, None
    for ln in query_lines:
        m = _FACT_RE.search(ln)
        if m:
            qpred, qval = m.group(1), m.group(3) == "True"
            break

    if entity is None or qpred is None:
        raise ValueError(f"Could not parse problem; missing fact/query: {raw[:200]}")

    return Problem(
        entity=entity, facts=frozenset(facts),
        rules=tuple(rules), query=(qpred, qval), raw=raw,
    )


def forward_apply(state: frozenset[tuple[str, bool]], rule: Rule
                   ) -> Optional[frozenset[tuple[str, bool]]]:
    """Apply rule to state. Returns new state, or None if no progress.

    Premise must be satisfied in state (exact match including value).
    Conclusion must not already be in state.
    """
    if (rule.premise_pred, rule.premise_val) not in state:
        return None
    new_fact = (rule.conclusion_pred, rule.conclusion_val)
    if new_fact in state:
        return None
    # Disallow contradictions: if (P, not value) is already in state, skip.
    contradicting = (rule.conclusion_pred, not rule.conclusion_val)
    if contradicting in state:
        return None
    return state | {new_fact}


def decidable(state: frozenset[tuple[str, bool]],
              query: tuple[str, bool]) -> bool:
    """True iff the state contains the query predicate (with either value).

    Once we know `Sour(x, True)` or `Sour(x, False)`, the answer is determined.
    """
    qpred, _ = query
    return any(p == qpred for (p, _) in state)


# --- Tree enumeration ---


@dataclass
class Node:
    node_id: int
    state: frozenset[tuple[str, bool]]
    parent: Optional[int]
    rule_used: Optional[Rule]   # the rule that was applied at the parent to reach this state
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_decidable: bool = False
    v_value: int = -1            # BFS distance to nearest decidable; -1 if unreachable


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


def enumerate_tree(problem: Problem, max_nodes: int = 5000) -> Tree:
    """BFS over reachable states. Branches by which rule to apply.

    Stops at decidable states (don't expand further) or max_nodes cap.
    """
    nodes: list[Node] = []
    seen: dict = {}   # state → node_id (first occurrence)

    root = Node(
        node_id=0, state=problem.facts, parent=None, rule_used=None, depth=0,
        is_decidable=decidable(problem.facts, problem.query),
    )
    nodes.append(root)
    seen[root.state] = 0

    frontier = [0]
    while frontier and len(nodes) < max_nodes:
        next_frontier = []
        for pid in frontier:
            parent = nodes[pid]
            if parent.is_decidable:
                continue
            for rule in problem.rules:
                new_state = forward_apply(parent.state, rule)
                if new_state is None:
                    continue
                if new_state in seen:
                    nid = seen[new_state]
                    if nid not in parent.children:
                        parent.children.append(nid)
                    continue
                nid = len(nodes)
                child = Node(
                    node_id=nid, state=new_state, parent=pid, rule_used=rule,
                    depth=parent.depth + 1,
                    is_decidable=decidable(new_state, problem.query),
                )
                nodes.append(child)
                seen[new_state] = nid
                parent.children.append(nid)
                if not child.is_decidable:
                    next_frontier.append(nid)
                if len(nodes) >= max_nodes:
                    break
            if len(nodes) >= max_nodes:
                break
        frontier = next_frontier

    # Compute v-values: BFS from decidable nodes (over the *undirected* edges
    # parent↔child in this tree).
    decidable_ids = [n.node_id for n in nodes if n.is_decidable]
    if not decidable_ids:
        # No decidable state found — leave v=-1 everywhere.
        return Tree(problem=problem, nodes=nodes)

    # build undirected adjacency
    adj: dict[int, list[int]] = {n.node_id: [] for n in nodes}
    for n in nodes:
        if n.parent is not None:
            adj[n.parent].append(n.node_id)
            adj[n.node_id].append(n.parent)

    from collections import deque
    dist: dict[int, int] = {nid: 0 for nid in decidable_ids}
    q = deque(decidable_ids)
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    for n in nodes:
        n.v_value = dist.get(n.node_id, -1)

    return Tree(problem=problem, nodes=nodes)


def render_state(problem: Problem, state: frozenset[tuple[str, bool]]) -> str:
    """Render a state as natural-language text, similar to the eval prompt."""
    parts = []
    parts.append(f"Initial fact: {problem.entity} is "
                 f"{'a' if next(iter(problem.facts))[1] else 'not a'} "
                 f"{next(iter(problem.facts))[0].lower()}.")
    derived = sorted(state - problem.facts)
    if derived:
        parts.append("Derived so far:")
        for pred, val in derived:
            verb = "is" if val else "is not"
            parts.append(f"  {problem.entity} {verb} {pred.lower()}.")
    qpred, _ = problem.query
    parts.append(f"Question: is {problem.entity} {qpred.lower()}?")
    return "\n".join(parts)


if __name__ == "__main__":
    # Quick smoke test.
    from datasets import load_dataset
    ds = load_dataset("renma/ProntoQA", split="validation")
    for i in [0, 100, 250]:
        rec = ds[i]
        p = parse_problem(rec["raw_logic_programs"])
        tree = enumerate_tree(p)
        v_values = sorted({n.v_value for n in tree.nodes if n.v_value >= 0})
        n_decidable = sum(1 for n in tree.nodes if n.is_decidable)
        print(f"#{i} ({rec['id']}): {len(tree.nodes)} nodes, "
              f"{n_decidable} decidable, v range = {v_values[:6]}..{v_values[-3:]}, "
              f"max depth = {max(n.depth for n in tree.nodes)}")
        # show sample state rendering
        sample = tree.nodes[len(tree.nodes) // 2]
        print(f"   sample state v={sample.v_value}:")
        print("   " + render_state(p, sample.state).replace("\n", "\n   "))
        print()
