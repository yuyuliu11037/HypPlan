"""State tree for Countdown (variable-target, N=6).

Variable-target, larger-N analog of src/tree_data.py. Because N=6 makes the
full history-dependent tree combinatorially huge (>1M nodes), we build a
*sampled history subtree*: start from the root and sample n_trajectories random
legal trajectories; the union of sampled paths forms a tree in which each
distinct (parent, op) edge reaches a distinct child node.

Node fields mirror src/tree_data.TreeNode (remaining, history, parent, depth,
is_terminal, is_success), plus `v_value` = BFS edge distance from this state
to the nearest success terminal, computed on the full state DAG (not the
sampled subtree) using the oracle's reachability cache.

Rendering mirrors src/tree_data.render_state with an added "Target: T" line
and integer formatting (no Fraction).

pair_distances_lca and parent_child_edges are reusable from src.tree_data
because they only depend on the parent/depth arrays.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


OPS = [
    ("+", lambda a, b: a + b, True),
    ("-", lambda a, b: a - b if a >= b else None, False),
    ("*", lambda a, b: a * b, True),
    ("/", lambda a, b: a // b if b != 0 and a % b == 0 else None, False),
]


@dataclass
class CDNode:
    node_id: int
    remaining: tuple              # tuple[int, ...]
    history: tuple                # tuple[(int, str, int, int), ...]
    parent: Optional[int]
    children: list = field(default_factory=list)
    depth: int = 0
    is_terminal: bool = False
    is_success: bool = False
    v_value: int = -1             # -1 = unreachable to success


@dataclass
class CDTree:
    pool: list
    target: int
    nodes: list

    @property
    def n(self) -> int:
        return len(self.nodes)


def legal_transitions(remaining: tuple) -> list:
    """Return [(a, sym, b, r, new_remaining), ...] unique by canonical key."""
    out = []
    seen: set = set()
    n = len(remaining)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = remaining[i], remaining[j]
            rest = tuple(remaining[k] for k in range(n) if k != i and k != j)
            for sym, fn, commutative in OPS:
                if commutative and a > b:
                    continue
                r = fn(a, b)
                if r is None:
                    continue
                new_rem = tuple(sorted(rest + (r,)))
                key = (sym, a, b, new_rem)
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, sym, b, r, new_rem))
    return out


def compute_v_values(oracle_cache: dict, target: int) -> dict:
    """For every state in oracle_cache (and its size-1 successors), compute
    v(state) = shortest number of legal ops to reach a success terminal.
    Unreachable states get v = -1.

    Size-1 terminals are NOT in oracle_cache (oracle_cd.can_reach early-returns
    before caching), so we enumerate them on the fly from size-2 successors.
    """
    v: dict[tuple, int] = {}

    # Discover size-1 terminals as successors of any size-2 state
    for s in oracle_cache:
        if len(s) == 2:
            for _a, _sym, _b, _r, new_rem in legal_transitions(s):
                if len(new_rem) == 1 and new_rem not in v:
                    v[new_rem] = 0 if new_rem[0] == target else -1

    # Initialize non-terminals as unknown (will be filled below)
    for s in oracle_cache:
        if s not in v:
            v[s] = -1

    # Bottom-up pass by ascending size
    states_by_size: dict[int, list] = {}
    for s in v:
        states_by_size.setdefault(len(s), []).append(s)

    for size in sorted(states_by_size.keys()):
        if size == 1:
            continue  # already set
        for s in states_by_size[size]:
            best = -1
            for _a, _sym, _b, _r, new_rem in legal_transitions(s):
                nv = v.get(new_rem, -1)
                if nv != -1 and (best == -1 or nv < best):
                    best = nv
            v[s] = -1 if best == -1 else best + 1
    return v


def sample_tree(pool: list[int], target: int, oracle_cache: dict,
                n_trajectories: int, n_guided: int = 0,
                seed: int = 0) -> CDTree:
    """Build a history-subtree by sampling trajectories through the state DAG.

    The first `n_guided` trajectories pick only winning ops (v-decreasing); the
    remaining `n_trajectories - n_guided` are uniform-random over legal ops.
    Same (parent, op) edge reuses the child node, so tree size grows
    sub-linearly in n_trajectories.

    Guided trajectories guarantee success terminals and good coverage of
    on-solution states; random trajectories fill in dead-end / off-path
    diversity needed for the head to discriminate.
    """
    rng = random.Random(seed)
    root_remaining = tuple(sorted(pool))
    v_values = compute_v_values(oracle_cache, target)

    nodes: list[CDNode] = []
    edge_to_child: dict[tuple, int] = {}

    root = CDNode(
        node_id=0, remaining=root_remaining, history=tuple(), parent=None,
        depth=0,
        is_terminal=len(root_remaining) == 1,
        is_success=(len(root_remaining) == 1 and root_remaining[0] == target),
        v_value=v_values.get(root_remaining, -1),
    )
    nodes.append(root)

    max_depth = len(pool) - 1
    for t in range(n_trajectories):
        guided = t < n_guided
        cur_id = 0
        for _ in range(max_depth):
            cur = nodes[cur_id]
            if cur.is_terminal:
                break
            transitions = legal_transitions(cur.remaining)
            if not transitions:
                break
            if guided:
                winning = [tr for tr in transitions
                           if v_values.get(tr[4], -1) != -1
                           and v_values[tr[4]] < v_values.get(cur.remaining, 999)]
                if not winning:
                    break  # current state has no winning successor
                a, sym, b, r, new_rem = rng.choice(winning)
            else:
                a, sym, b, r, new_rem = rng.choice(transitions)
            edge_key = (cur_id, sym, a, b, r)
            if edge_key in edge_to_child:
                cur_id = edge_to_child[edge_key]
                continue
            new_node = CDNode(
                node_id=len(nodes),
                remaining=new_rem,
                history=cur.history + ((a, sym, b, r),),
                parent=cur_id,
                depth=cur.depth + 1,
                is_terminal=(len(new_rem) == 1),
                is_success=(len(new_rem) == 1 and new_rem[0] == target),
                v_value=v_values.get(new_rem, -1),
            )
            nodes.append(new_node)
            nodes[cur_id].children.append(new_node.node_id)
            edge_to_child[edge_key] = new_node.node_id
            cur_id = new_node.node_id

    return CDTree(pool=list(pool), target=target, nodes=nodes)


def render_state(tree: CDTree, node: CDNode) -> str:
    """Canonical text rendering. Matches the Countdown SFT trajectory format."""
    pool_str = " ".join(str(n) for n in sorted(tree.pool))
    lines = [f"Problem: {pool_str} | Target: {tree.target}"]
    working = list(sorted(tree.pool))
    hist = node.history
    for i, (a, op, b, r) in enumerate(hist):
        working.remove(a)
        working.remove(b)
        working.append(r)
        is_last = i == len(hist) - 1
        if is_last and len(working) == 1 and r == tree.target:
            lines.append(f"Step {i+1}: {a} {op} {b} = {r}. Answer: {tree.target}")
        else:
            rem_str = " ".join(str(x) for x in sorted(working))
            lines.append(f"Step {i+1}: {a} {op} {b} = {r}. Remaining: {rem_str}")
    if len(hist) == 0:
        rem_str = " ".join(str(x) for x in sorted(node.remaining))
        lines.append(f"Remaining: {rem_str}")
    return "\n".join(lines)


def render_state_from_history(pool, target, history) -> str:
    """Render without a Tree object. Defensive: truncates at first invalid op."""
    pool_str = " ".join(str(n) for n in sorted(pool))
    lines = [f"Problem: {pool_str} | Target: {target}"]
    working = list(sorted(pool))
    for i, (a, op, b, r) in enumerate(history):
        if a not in working:
            break
        working.remove(a)
        if b not in working:
            working.append(a)
            break
        working.remove(b)
        working.append(r)
        is_last = i == len(history) - 1
        if is_last and len(working) == 1 and r == target:
            lines.append(f"Step {i+1}: {a} {op} {b} = {r}. Answer: {target}")
        else:
            rem_str = " ".join(str(x) for x in sorted(working))
            lines.append(f"Step {i+1}: {a} {op} {b} = {r}. Remaining: {rem_str}")
    if len(history) == 0:
        rem_str = " ".join(str(x) for x in sorted(pool))
        lines.append(f"Remaining: {rem_str}")
    return "\n".join(lines)


def parent_child_edges(tree: CDTree) -> np.ndarray:
    edges = [(n.parent, n.node_id) for n in tree.nodes if n.parent is not None]
    return (np.array(edges, dtype=np.int32) if edges
            else np.zeros((0, 2), dtype=np.int32))


def non_descendants(tree: CDTree, node_id: int) -> list[int]:
    subtree: set = set()
    stack = [node_id]
    while stack:
        u = stack.pop()
        subtree.add(u)
        stack.extend(tree.nodes[u].children)
    return [i for i in range(tree.n) if i not in subtree]
