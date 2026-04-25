"""Blocksworld oracle: state representation, action application, BFS to goal.

State representation: a frozenset of facts. Each fact is one of:
  ("on", X, Y)         — block X is on block Y
  ("ontable", X)       — block X is on the table
  ("clear", X)         — block X has nothing on top of it
  ("holding", X)       — the hand is holding block X
  ("handempty",)       — the hand is empty

Actions (the standard 4-op blocksworld domain):
  pick-up(X)           — from table; preconds: ontable(X), clear(X), handempty
  put-down(X)          — to table; preconds: holding(X)
  stack(X, Y)          — onto Y; preconds: holding(X), clear(Y)
  unstack(X, Y)        — from on Y; preconds: on(X,Y), clear(X), handempty

Public API:
- parse_problem(prompt: str) -> Problem (extracts initial state + goal from
  the trailing [STATEMENT] block of a PlanBench query).
- applicable_actions(state) -> list[Action]
- apply_action(state, action) -> new_state
- is_goal(state, goal) -> bool (every goal fact is in state)
- enumerate_tree(problem, max_nodes) -> Tree (BFS, marks goal-reaching nodes)
- render_state(state) -> human-readable text
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


_BLOCK_RE = r"\b(\w+ block)\b"


@dataclass(frozen=True)
class Action:
    op: str                   # "pick-up" | "put-down" | "stack" | "unstack"
    args: tuple[str, ...]

    def render(self) -> str:
        return f"({self.op} {' '.join(self.args)})"


@dataclass
class Problem:
    blocks: tuple[str, ...]
    init: frozenset                # set of fact tuples
    goal: frozenset                # subset of facts the goal requires
    raw: str = ""


# Parse the natural-language [STATEMENT] block at the end of a PlanBench
# query into structured facts. Patterns are robust to phrasing variations.
def _extract_blocks(text: str) -> set[str]:
    return set(m.group(1).split()[0] for m in re.finditer(_BLOCK_RE, text))


def _parse_state_phrase(text: str) -> set[tuple]:
    facts = set()
    for m in re.finditer(r"the (\w+) block is clear", text):
        facts.add(("clear", m.group(1)))
    for m in re.finditer(r"the hand is empty", text):
        facts.add(("handempty",))
    for m in re.finditer(r"the (\w+) block is on top of the (\w+) block", text):
        facts.add(("on", m.group(1), m.group(2)))
    for m in re.finditer(r"the (\w+) block is on the table", text):
        facts.add(("ontable", m.group(1)))
    for m in re.finditer(r"holding the (\w+) block", text):
        facts.add(("holding", m.group(1)))
    return facts


def parse_problem(prompt: str) -> Problem:
    """Extract the LAST [STATEMENT] in the prompt (the actual problem)."""
    parts = prompt.split("[STATEMENT]")
    last = parts[-1]
    # Split on "My goal is to have that"
    if "My goal is to have that" not in last:
        raise ValueError("No goal phrase found")
    init_text, goal_text = last.split("My goal is to have that", 1)
    init = _parse_state_phrase(init_text)
    # goal: stop at "My plan is" or end of statement
    goal_text = goal_text.split("My plan is")[0]
    goal = _parse_state_phrase(goal_text)
    blocks = sorted(_extract_blocks(init_text + goal_text))
    return Problem(
        blocks=tuple(blocks), init=frozenset(init), goal=frozenset(goal),
        raw=last,
    )


# --- Action semantics ---


def applicable_actions(state: frozenset, blocks: tuple[str, ...]
                        ) -> list[Action]:
    out: list[Action] = []
    handempty = ("handempty",) in state
    holding = next((f[1] for f in state if f[0] == "holding"), None)
    for b in blocks:
        ontable = ("ontable", b) in state
        clear = ("clear", b) in state
        # pick-up
        if handempty and ontable and clear:
            out.append(Action("pick-up", (b,)))
        # put-down
        if holding == b:
            out.append(Action("put-down", (b,)))
        for b2 in blocks:
            if b == b2:
                continue
            # stack b onto b2
            if holding == b and ("clear", b2) in state:
                out.append(Action("stack", (b, b2)))
            # unstack b from b2
            if (handempty and ("on", b, b2) in state
                and ("clear", b) in state):
                out.append(Action("unstack", (b, b2)))
    return out


def apply_action(state: frozenset, a: Action) -> frozenset:
    s = set(state)
    if a.op == "pick-up":
        (b,) = a.args
        s.discard(("ontable", b))
        s.discard(("clear", b))
        s.discard(("handempty",))
        s.add(("holding", b))
    elif a.op == "put-down":
        (b,) = a.args
        s.discard(("holding", b))
        s.add(("ontable", b))
        s.add(("clear", b))
        s.add(("handempty",))
    elif a.op == "stack":
        b, b2 = a.args
        s.discard(("holding", b))
        s.discard(("clear", b2))
        s.add(("on", b, b2))
        s.add(("clear", b))
        s.add(("handempty",))
    elif a.op == "unstack":
        b, b2 = a.args
        s.discard(("on", b, b2))
        s.discard(("clear", b))
        s.discard(("handempty",))
        s.add(("holding", b))
        s.add(("clear", b2))
    return frozenset(s)


def is_goal(state: frozenset, goal: frozenset) -> bool:
    return goal.issubset(state)


# --- Tree enumeration (BFS to optimal-distance goal) ---


@dataclass
class Node:
    node_id: int
    state: frozenset
    parent: Optional[int]
    action_used: Optional[Action]
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_goal: bool = False
    v_value: int = -1


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


def enumerate_tree(problem: Problem, max_nodes: int = 5000) -> Tree:
    """Forward BFS from init, branching on every applicable action.

    Stops expanding goal-reaching states. Caps total node count.
    """
    nodes: list[Node] = []
    seen: dict = {}
    root = Node(node_id=0, state=problem.init, parent=None, action_used=None,
                 depth=0, is_goal=is_goal(problem.init, problem.goal))
    nodes.append(root)
    seen[root.state] = 0
    frontier = [0]
    while frontier and len(nodes) < max_nodes:
        next_frontier = []
        for pid in frontier:
            parent = nodes[pid]
            if parent.is_goal:
                continue
            for action in applicable_actions(parent.state, problem.blocks):
                new_state = apply_action(parent.state, action)
                if new_state in seen:
                    nid = seen[new_state]
                    if nid not in parent.children:
                        parent.children.append(nid)
                    continue
                nid = len(nodes)
                child = Node(node_id=nid, state=new_state, parent=pid,
                              action_used=action, depth=parent.depth + 1,
                              is_goal=is_goal(new_state, problem.goal))
                nodes.append(child)
                seen[new_state] = nid
                parent.children.append(nid)
                if not child.is_goal:
                    next_frontier.append(nid)
                if len(nodes) >= max_nodes:
                    break
            if len(nodes) >= max_nodes:
                break
        frontier = next_frontier

    # v-value = BFS distance to goal (over undirected parent↔child edges)
    goal_ids = [n.node_id for n in nodes if n.is_goal]
    if goal_ids:
        adj: dict = {n.node_id: [] for n in nodes}
        for n in nodes:
            if n.parent is not None:
                adj[n.parent].append(n.node_id)
                adj[n.node_id].append(n.parent)
        from collections import deque
        dist = {gid: 0 for gid in goal_ids}
        q = deque(goal_ids)
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        for n in nodes:
            n.v_value = dist.get(n.node_id, -1)
    return Tree(problem=problem, nodes=nodes)


# --- Rendering ---


def render_state(problem: Problem, state: frozenset) -> str:
    parts = []
    parts.append("Current state:")
    on_facts = sorted([(f[1], f[2]) for f in state if f[0] == "on"])
    table_facts = sorted([f[1] for f in state if f[0] == "ontable"])
    clear_facts = sorted([f[1] for f in state if f[0] == "clear"])
    if any(f[0] == "handempty" for f in state):
        parts.append("  the hand is empty")
    held = next((f[1] for f in state if f[0] == "holding"), None)
    if held:
        parts.append(f"  holding the {held} block")
    for top, bottom in on_facts:
        parts.append(f"  the {top} block is on top of the {bottom} block")
    for b in table_facts:
        parts.append(f"  the {b} block is on the table")
    for b in clear_facts:
        parts.append(f"  the {b} block is clear")

    parts.append("Goal:")
    g_on = sorted([(f[1], f[2]) for f in problem.goal if f[0] == "on"])
    g_table = sorted([f[1] for f in problem.goal if f[0] == "ontable"])
    for top, bottom in g_on:
        parts.append(f"  {top} on {bottom}")
    for b in g_table:
        parts.append(f"  {b} on the table")
    return "\n".join(parts)


if __name__ == "__main__":
    # Smoke test on a sample PlanBench problem.
    import json
    rec = json.loads(open("data/blocksworld_test.jsonl").readline())
    p = parse_problem(rec["prompt"])
    print("Blocks:", p.blocks)
    print("Init:", sorted(p.init)[:5], "...")
    print("Goal:", sorted(p.goal))
    print()
    tree = enumerate_tree(p)
    print(f"Tree: {len(tree.nodes)} nodes, "
           f"{sum(1 for n in tree.nodes if n.is_goal)} goal-reaching, "
           f"max depth = {max(n.depth for n in tree.nodes)}")
    v_values = sorted({n.v_value for n in tree.nodes if n.v_value >= 0})
    print(f"v range: {v_values[:6]} .. {v_values[-3:]}")
    sample = tree.nodes[len(tree.nodes) // 2]
    print(f"\nSample state v={sample.v_value}:")
    print(render_state(p, sample.state))
