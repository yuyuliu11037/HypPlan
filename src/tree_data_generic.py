"""Generic (varied-pool, varied-target) tree enumeration and state rendering.

Parallel to [src/tree_data.py](src/tree_data.py) but does not hardcode target=24.
Given any positive-integer pool and any positive-integer target, enumerates the
full state tree and provides canonical state text that matches our generic
prompt format (`Numbers: ... | Target: T` header).

State text used for Stage-1 head hidden-state caching. Target is included in
the header so the head can learn target-aware topology (states near
target-matching leaves are close in hyperbolic space).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Optional

from src.tree_data import OPS, fraction_to_str


@dataclass
class TreeNodeG:
    node_id: int
    remaining: tuple
    history: tuple
    parent: Optional[int]
    children: list = field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False
    is_success: bool = False


@dataclass
class TreeG:
    pool: tuple                 # tuple of Fraction
    target: int
    nodes: list

    @property
    def n(self) -> int:
        return len(self.nodes)


def enumerate_tree_generic(pool: list[int], target: int) -> TreeG:
    """Enumerate the tree for (pool, target). Success = terminal whose value
    equals target. Same per-frame dedup as `enumerate_tree`."""
    init = tuple(sorted(Fraction(int(x)) for x in pool))
    target_f = Fraction(int(target))

    nodes: list[TreeNodeG] = []
    root = TreeNodeG(
        node_id=0,
        remaining=init,
        history=tuple(),
        parent=None,
        depth=0,
        is_terminal=(len(init) == 1),
        is_success=(len(init) == 1 and init[0] == target_f),
    )
    nodes.append(root)

    stack: list[int] = [0]
    while stack:
        pid = stack.pop()
        parent = nodes[pid]
        if parent.is_terminal:
            continue

        rem = parent.remaining
        seen_pairs: set = set()
        for i, j in combinations(range(len(rem)), 2):
            a, b = rem[i], rem[j]
            leftover = tuple(rem[k] for k in range(len(rem)) if k != i and k != j)
            for op_sym, op_fn, commutative in OPS:
                orderings = [(a, b)] if commutative else [(a, b), (b, a)]
                for x, y in orderings:
                    result = op_fn(x, y)
                    if result is None:
                        continue
                    key = (op_sym, x, y, tuple(sorted(leftover)))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    new_rem = tuple(sorted(leftover + (result,)))
                    new_hist = parent.history + ((x, op_sym, y, result),)
                    child_id = len(nodes)
                    is_term = len(new_rem) == 1
                    child = TreeNodeG(
                        node_id=child_id,
                        remaining=new_rem,
                        history=new_hist,
                        parent=pid,
                        depth=parent.depth + 1,
                        is_terminal=is_term,
                        is_success=is_term and new_rem[0] == target_f,
                    )
                    nodes.append(child)
                    parent.children.append(child_id)
                    stack.append(child_id)

    return TreeG(pool=init, target=target, nodes=nodes)


def _header(pool: tuple, target: int) -> str:
    nums = " ".join(fraction_to_str(x) for x in sorted(pool))
    return f"Numbers: {nums} | Target: {int(target)}"


def render_state_generic(pool: list[int], target: int, history: tuple) -> str:
    """Render a generic-format state text.

    Defensive: if `history` references operands absent from the current pool,
    silently truncate at the last valid step.
    """
    pool_fr = [Fraction(int(x)) for x in pool]
    lines = [_header(tuple(Fraction(int(x)) for x in pool), target)]

    working = list(pool_fr)
    for i, (a, op, b, r) in enumerate(history):
        if a not in working:
            break
        working.remove(a)
        if b not in working:
            working.append(a)  # restore pool to pre-step view
            break
        working.remove(b)
        working.append(r)
        step_str = f"{fraction_to_str(a)} {op} {fraction_to_str(b)} = {fraction_to_str(r)}"
        is_last_step = (i == len(history) - 1)
        if is_last_step and len(working) == 1 and r == Fraction(int(target)):
            lines.append(f"Step {i+1}: {step_str}. Answer: {int(target)}")
        else:
            rem_str = " ".join(fraction_to_str(x) for x in sorted(working))
            lines.append(f"Step {i+1}: {step_str}. Remaining: {rem_str}")

    if len(history) == 0:
        rem_str = " ".join(fraction_to_str(x) for x in sorted(working))
        lines.append(f"Remaining: {rem_str}")
    return "\n".join(lines)


def render_tree_node(tree: TreeG, node: TreeNodeG) -> str:
    return render_state_generic([int(x) for x in tree.pool], tree.target,
                                node.history)


def bfs_distances_to_success(tree: TreeG) -> list[int]:
    """For each node, BFS distance (in edges) to the nearest success terminal.
    Unreachable nodes get distance 10**9 (sentinel)."""
    n = tree.n
    adj: list[list[int]] = [list(tree.nodes[i].children) for i in range(n)]
    # add reverse edges
    for i in range(n):
        for c in tree.nodes[i].children:
            adj[c].append(i)

    INF = 10 ** 9
    dist = [INF] * n
    from collections import deque
    q = deque()
    for i in range(n):
        if tree.nodes[i].is_success:
            dist[i] = 0
            q.append(i)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
