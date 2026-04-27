"""Number-path / reachability oracle (candidate Group A OOD #1).

Given a start integer S, a target integer T, and a small set of allowed
arithmetic operations, find a sequence of operations that transforms S
into T. Each problem fixes the op set; the model must search over which
op to apply at each step.

Op vocabulary (for generation):
    +a, -a   for a ∈ {1, 2, 3, 5, 7}
    *k, //k  for k ∈ {2, 3}
Each op is encoded as ("ADD"|"SUB"|"MUL"|"DIV", const).

State = current integer. Action = one op. Goal = state == target.

Operations have validity constraints:
    - SUB: result must be non-negative.
    - DIV: result must be an exact integer; divisor != 0.
    - MUL/ADD: result must stay within the bounded search range.

Tree search uses BFS bounded by `max_value` and `max_depth`. The oracle
exposes the standard Tree/Node + winning_steps interface for HypPlan.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------- Op definitions ----------------------------

OP_KINDS = ("ADD", "SUB", "MUL", "DIV")


@dataclass(frozen=True)
class Op:
    kind: str            # ADD/SUB/MUL/DIV
    const: int

    def render(self) -> str:
        if self.kind == "ADD":
            return f"+ {self.const}"
        if self.kind == "SUB":
            return f"- {self.const}"
        if self.kind == "MUL":
            return f"* {self.const}"
        if self.kind == "DIV":
            return f"/ {self.const}"
        return "?"

    def apply(self, x: int, max_value: int = 999) -> Optional[int]:
        if self.kind == "ADD":
            r = x + self.const
        elif self.kind == "SUB":
            r = x - self.const
            if r < 0:
                return None
        elif self.kind == "MUL":
            r = x * self.const
        elif self.kind == "DIV":
            if self.const == 0 or x % self.const != 0:
                return None
            r = x // self.const
        else:
            return None
        if r < 0 or r > max_value:
            return None
        return r


# ---------------------------- Data classes ----------------------------

@dataclass
class Problem:
    start: int
    target: int
    ops: tuple[Op, ...]
    max_value: int = 999
    raw: str = ""

    def render_problem(self) -> str:
        op_str = ", ".join(o.render() for o in self.ops)
        return (
            f"Apply operations to transform the start number into the target "
            f"number. Use any sequence of the allowed operations.\n"
            f"Start: {self.start}    Target: {self.target}\n"
            f"Operations: {op_str}"
        )


@dataclass
class Node:
    node_id: int
    state: int
    parent: Optional[int]
    op_used: Optional[Op]
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_solved: bool = False
    v_value: int = -1


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


# ---------------------------- Oracle ----------------------------

def is_solved(state: int, target: int) -> bool:
    return state == target


def applicable_ops(state: int, problem: Problem) -> list[Op]:
    out: list[Op] = []
    for op in problem.ops:
        if op.apply(state, problem.max_value) is not None:
            out.append(op)
    return out


def apply_op(state: int, op: Op, problem: Problem) -> Optional[int]:
    return op.apply(state, problem.max_value)


def enumerate_tree(problem: Problem, max_nodes: int = 5000,
                    max_depth: int = 12) -> Tree:
    root = Node(
        node_id=0, state=problem.start, parent=None, op_used=None,
        depth=0, is_solved=is_solved(problem.start, problem.target),
    )
    nodes: list[Node] = [root]
    seen: dict[int, int] = {problem.start: 0}
    queue: deque[int] = deque([0])
    while queue and len(nodes) < max_nodes:
        nid = queue.popleft()
        node = nodes[nid]
        if node.is_solved or node.depth >= max_depth:
            continue
        for op in applicable_ops(node.state, problem):
            ns = apply_op(node.state, op, problem)
            if ns is None:
                continue
            if ns in seen:
                cid = seen[ns]
            else:
                cid = len(nodes)
                nodes.append(Node(
                    node_id=cid, state=ns, parent=nid, op_used=op,
                    depth=node.depth + 1,
                    is_solved=is_solved(ns, problem.target),
                ))
                seen[ns] = cid
                queue.append(cid)
                if len(nodes) >= max_nodes:
                    break
            if cid not in node.children:
                node.children.append(cid)

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


def winning_steps(state: int, problem: Problem,
                   max_nodes: int = 5000) -> list[Op]:
    """Ops in `state` that strictly decrease BFS distance to target."""
    if is_solved(state, problem.target):
        return []

    def min_dist(start: int) -> Optional[int]:
        if is_solved(start, problem.target):
            return 0
        seen: dict[int, int] = {start: 0}
        q: deque[int] = deque([start])
        while q and len(seen) < max_nodes:
            cur = q.popleft()
            d = seen[cur]
            for op in applicable_ops(cur, problem):
                ns = apply_op(cur, op, problem)
                if ns is None or ns in seen:
                    continue
                nd = d + 1
                if is_solved(ns, problem.target):
                    return nd
                seen[ns] = nd
                q.append(ns)
        return None

    s_dist = min_dist(state)
    if s_dist is None:
        return []
    out: list[Op] = []
    for op in applicable_ops(state, problem):
        ns = apply_op(state, op, problem)
        if ns is None:
            continue
        nd = min_dist(ns)
        if nd is not None and nd == s_dist - 1:
            out.append(op)
    return out


def validate_step(state: int, op: Op,
                   problem: Problem) -> tuple[bool, int]:
    if op not in problem.ops:
        return False, state
    ns = op.apply(state, problem.max_value)
    if ns is None:
        return False, state
    return True, ns


# ---------------------------- Problem generation ----------------------------

OP_BANK = (
    Op("ADD", 1), Op("ADD", 2), Op("ADD", 3), Op("ADD", 5), Op("ADD", 7),
    Op("SUB", 1), Op("SUB", 2), Op("SUB", 3), Op("SUB", 5), Op("SUB", 7),
    Op("MUL", 2), Op("MUL", 3),
    Op("DIV", 2), Op("DIV", 3),
)


def generate_problem(target_depth: int, op_set_size: int = 4,
                      max_value: int = 999, seed: int = 0,
                      max_attempts: int = 100) -> Problem:
    """Generate a problem solvable in EXACTLY `target_depth` steps.

    Process:
    1. Pick `op_set_size` distinct ops from OP_BANK.
    2. Pick a start S in [1, 50].
    3. Random-walk `target_depth` steps to get a candidate target T.
    4. Verify the BFS minimum distance from S to T is exactly target_depth
       (not shorter via a different path).
    """
    rng = random.Random(seed)
    for _ in range(max_attempts):
        ops = tuple(rng.sample(OP_BANK, op_set_size))
        start = rng.randint(1, 50)
        problem = Problem(start=start, target=start, ops=ops,
                           max_value=max_value)

        cur = start
        for _ in range(target_depth):
            applicable = applicable_ops(cur, problem)
            if not applicable:
                cur = None
                break
            op = rng.choice(applicable)
            cur = apply_op(cur, op, problem)
        if cur is None or cur == start:
            continue
        target = cur
        problem = Problem(start=start, target=target, ops=ops,
                           max_value=max_value)
        # Verify minimum distance == target_depth.
        tree = enumerate_tree(problem, max_nodes=2000,
                               max_depth=target_depth + 2)
        root_d = tree.nodes[0].v_value
        if root_d == target_depth:
            return problem
    raise RuntimeError(
        f"Could not generate number-path problem at depth={target_depth} "
        f"after {max_attempts} attempts"
    )


# ---------------------------- Rendering / parsing ----------------------------

def render_state(problem: Problem, state: int) -> str:
    op_str = ", ".join(o.render() for o in problem.ops)
    return (
        f"Current value: {state}. Target: {problem.target}. "
        f"Operations: {op_str}"
    )


def format_question(problem: Problem) -> str:
    return problem.render_problem()


def format_step_text(state_before: int, op: Op, state_after: int,
                      step_num: int) -> str:
    return f"{state_before} {op.render()} = {state_after}"


def format_gold_trajectory(problem: Problem, max_steps: int = 12) -> str:
    state = problem.start
    lines: list[str] = []
    step = 1
    while not is_solved(state, problem.target) and step <= max_steps:
        wins = winning_steps(state, problem)
        if not wins:
            break
        op = wins[0]
        ns = op.apply(state, problem.max_value)
        if ns is None:
            break
        lines.append(f"Step {step}: {format_step_text(state, op, ns, step)}")
        state = ns
        step += 1
    lines.append(f"Answer: {problem.target}")
    return "\n".join(lines)


# ---------------------------- Step parsing ----------------------------

import re

_STEP_RE = re.compile(
    r"(\d+)\s*([\+\-\*xX/])\s*(\d+)\s*=\s*(\d+)", re.IGNORECASE
)


def parse_step(text: str, problem: Problem,
                state: int) -> Optional[tuple[Op, int]]:
    """Find one 'a op b = r' assertion that's legal in the current state."""
    for m in _STEP_RE.finditer(text):
        a = int(m.group(1)); sym = m.group(2).lower(); b = int(m.group(3))
        r = int(m.group(4))
        if a != state:
            continue
        kind = {"+": "ADD", "-": "SUB", "*": "MUL", "x": "MUL",
                "/": "DIV"}.get(sym)
        if kind is None:
            continue
        op = Op(kind=kind, const=b)
        if op not in problem.ops:
            continue
        ns = op.apply(state, problem.max_value)
        if ns is None or ns != r:
            continue
        return op, ns
    return None


def parse_answer(text: str) -> Optional[int]:
    m = re.search(r"Answer\s*[:\-]?\s*(-?\d+)", text, re.IGNORECASE)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def score_answer(prediction: Optional[int], gold: int) -> bool:
    return prediction is not None and prediction == gold
