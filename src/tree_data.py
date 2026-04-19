"""Full-enumeration state trees for Game-of-24.

Each node is a state = (remaining numbers, ordered history of ops) in the search
tree. Nodes are history-dependent: the same `remaining` reached by two different
histories are two distinct nodes. Tree distance = edge count on the undirected
tree.

Reused from data/generate_24_trajectories.py: OPS table, per-frame canonical
dedup, fraction_to_str formatter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Optional

import numpy as np


OPS = [
    ("+", lambda a, b: a + b, True),
    ("-", lambda a, b: a - b, False),
    ("*", lambda a, b: a * b, True),
    ("/", lambda a, b: a / b if b != 0 else None, False),
]


def fraction_to_str(f: Fraction) -> str:
    return str(int(f)) if f.denominator == 1 else str(f)


@dataclass
class TreeNode:
    node_id: int
    remaining: tuple              # tuple[Fraction, ...]
    history: tuple                # tuple[(Fraction a, str op, Fraction b, Fraction r), ...]
    parent: Optional[int]
    children: list = field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False     # len(remaining) == 1
    is_success: bool = False      # terminal AND result == 24


@dataclass
class Tree:
    problem: str                  # "4,7,8,8"
    nodes: list                   # list[TreeNode]

    @property
    def n(self) -> int:
        return len(self.nodes)


def enumerate_tree(problem: str) -> Tree:
    """Enumerate the full state tree for a Game-of-24 problem.

    Problem is a string like "4,7,8,8". Root = (sorted numbers, empty history).
    Children = all (a, op, b) applications whose canonical key hasn't been seen
    at this frame.
    """
    init = tuple(sorted(Fraction(int(x)) for x in problem.split(",")))
    nodes: list[TreeNode] = []

    root = TreeNode(
        node_id=0,
        remaining=init,
        history=tuple(),
        parent=None,
        depth=0,
        is_terminal=(len(init) == 1),
        is_success=(len(init) == 1 and init[0] == Fraction(24)),
    )
    nodes.append(root)

    # Iterative DFS so we control ordering and per-frame dedup
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
                    child = TreeNode(
                        node_id=child_id,
                        remaining=new_rem,
                        history=new_hist,
                        parent=pid,
                        depth=parent.depth + 1,
                        is_terminal=is_term,
                        is_success=is_term and new_rem[0] == Fraction(24),
                    )
                    nodes.append(child)
                    parent.children.append(child_id)
                    stack.append(child_id)

    return Tree(problem=problem, nodes=nodes)


def render_state_from_history(problem: str, history: tuple) -> str:
    """Render canonical state text given the problem string and op history.

    Same format as render_state() but doesn't require a Tree/TreeNode object.
    Useful during stage-2 training where we walk through a trajectory step by
    step and need the canonical state text at each boundary.

    Defensive against invalid history: if a step references an operand not in
    the current pool (e.g. model emitted a hallucinated number), the history
    is silently truncated at the last valid step. The returned text reflects
    the longest valid prefix.
    """
    problem_nums = " ".join(fraction_to_str(Fraction(int(x)))
                            for x in problem.split(","))
    lines = [f"Problem: {problem_nums}"]
    pool = [Fraction(int(x)) for x in problem.split(",")]
    valid_history = []
    for i, (a, op, b, r) in enumerate(history):
        if a not in pool:
            break
        pool.remove(a)
        if b not in pool:
            pool.append(a)  # restore so output reflects state before this step
            break
        pool.remove(b)
        pool.append(r)
        valid_history.append((a, op, b, r))
        pool_sorted = sorted(pool)
        step_str = f"{fraction_to_str(a)} {op} {fraction_to_str(b)} = {fraction_to_str(r)}"
        is_last = (i == len(history) - 1)
        if is_last and len(pool) == 1 and r == Fraction(24):
            lines.append(f"Step {i+1}: {step_str}. Answer: 24")
        else:
            remaining_str = " ".join(fraction_to_str(x) for x in pool_sorted)
            lines.append(f"Step {i+1}: {step_str}. Remaining: {remaining_str}")
    if len(history) == 0:
        remaining_str = " ".join(fraction_to_str(x) for x in sorted(pool))
        lines.append(f"Remaining: {remaining_str}")
    return "\n".join(lines)


def render_state(tree: Tree, node: TreeNode) -> str:
    """Canonical text rendering of a state. Matches the SFT trajectory format.

    Root:
        Problem: 4 7 8 8
        Remaining: 4 7 8 8

    Intermediate (after k < 3 steps):
        Problem: 4 7 8 8
        Step 1: 8 / 8 = 1. Remaining: 1 4 7

    Terminal (k == 3):
        Problem: 4 7 8 8
        Step 1: 8 / 8 = 1. Remaining: 1 4 7
        Step 2: 7 - 1 = 6. Remaining: 4 6
        Step 3: 4 * 6 = 24. Answer: 24        (if success)
        Step 3: 4 * 6 = 23. Remaining: 23     (if failure — last 'Remaining' is single num)
    """
    problem_nums = " ".join(fraction_to_str(Fraction(int(x)))
                            for x in tree.problem.split(","))
    lines = [f"Problem: {problem_nums}"]

    pool = list(tree.nodes[0].remaining)  # start from root remaining
    hist = node.history
    for i, (a, op, b, r) in enumerate(hist):
        pool.remove(a)
        pool.remove(b)
        pool.append(r)
        pool_sorted = sorted(pool)
        step_str = f"{fraction_to_str(a)} {op} {fraction_to_str(b)} = {fraction_to_str(r)}"
        is_last = (i == len(hist) - 1)
        if is_last and len(pool) == 1 and r == Fraction(24):
            lines.append(f"Step {i+1}: {step_str}. Answer: 24")
        else:
            remaining_str = " ".join(fraction_to_str(x) for x in pool_sorted)
            lines.append(f"Step {i+1}: {step_str}. Remaining: {remaining_str}")

    if len(hist) == 0:
        remaining_str = " ".join(fraction_to_str(x) for x in node.remaining)
        lines.append(f"Remaining: {remaining_str}")

    return "\n".join(lines)


def tree_distance_matrix(tree: Tree) -> np.ndarray:
    """All-pairs shortest-path distances on the undirected tree. int16 matrix.

    BFS from each node. Tree has O(N) edges so each BFS is O(N); total O(N^2).
    Only used for eval; training uses pair_distances_lca() instead.
    """
    n = tree.n
    adj: list[list[int]] = [[] for _ in range(n)]
    for node in tree.nodes:
        if node.parent is not None:
            adj[node.parent].append(node.node_id)
            adj[node.node_id].append(node.parent)

    dist = np.full((n, n), -1, dtype=np.int16)
    for src in range(n):
        dist[src, src] = 0
        frontier = [src]
        while frontier:
            next_frontier = []
            for u in frontier:
                d_u = dist[src, u]
                for v in adj[u]:
                    if dist[src, v] == -1:
                        dist[src, v] = d_u + 1
                        next_frontier.append(v)
            frontier = next_frontier
    return dist


def pair_distances_lca(parents: np.ndarray, depths: np.ndarray,
                        i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Vectorized tree distance for K pairs via LCA.

    Args:
        parents: (N,) int array, parents[root] = -1
        depths: (N,) int array
        i, j: (K,) int arrays of node ids

    Returns:
        (K,) int array, d(i, j) = depth[i] + depth[j] - 2 * depth[LCA(i, j)]

    Since Game-of-24 trees have depth <= 4 (root + at most 4 ops? actually 3 for
    4-number problems since each op reduces pool size by 1), the climb loop is
    tiny. Implemented with Python-level iteration over K pairs — K is small
    (thousands) and parent chains are short.
    """
    K = i.shape[0]
    out = np.empty(K, dtype=np.int32)
    for k in range(K):
        u, v = int(i[k]), int(j[k])
        du, dv = int(depths[u]), int(depths[v])
        # Bring both to the same depth
        while du > dv:
            u = int(parents[u]); du -= 1
        while dv > du:
            v = int(parents[v]); dv -= 1
        # Walk up together
        while u != v:
            u = int(parents[u]); v = int(parents[v])
            du -= 1
        lca_depth = du
        out[k] = (int(depths[int(i[k])]) + int(depths[int(j[k])])
                  - 2 * lca_depth)
    return out


def parent_child_edges(tree: Tree) -> np.ndarray:
    """(E, 2) int32 array of (parent, child) edges. Useful for ranking loss."""
    edges = [(n.parent, n.node_id) for n in tree.nodes if n.parent is not None]
    return np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)


def non_descendants(tree: Tree, node_id: int) -> list[int]:
    """All node ids that are NOT in the subtree rooted at `node_id`.

    Used as the negative pool for Nickel-Kiela ranking.
    """
    subtree: set = set()
    stack = [node_id]
    while stack:
        u = stack.pop()
        subtree.add(u)
        stack.extend(tree.nodes[u].children)
    return [i for i in range(tree.n) if i not in subtree]
