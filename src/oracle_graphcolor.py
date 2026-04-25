"""Graph 3-coloring oracle: generate problems, enumerate state trees,
render states, score colorings.

Task: given an undirected graph G=(V,E), assign each vertex one of 3 colors
{R, G, B} such that no edge connects two same-colored vertices.

State = tuple of (vertex_id, color) for vertices already colored, in
canonical (sorted-by-vertex-id) order. An action picks the next uncolored
vertex (canonical order = lowest unfilled id) and assigns a non-conflicting
color. Tree branches over color choice at each step.

Public API:
- generate_problem(n, density, rng) -> Problem
- enumerate_tree(problem, max_nodes) -> Tree
- render_state(problem, state) -> str
- format_question(problem) -> str  (for eval prompts)
- format_gold_trajectory(problem, coloring) -> str (for SFT data)
- score_coloring(problem, coloring) -> bool
- parse_coloring(generation, problem) -> dict[int, str] | None
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional


COLORS = ("R", "G", "B")
COLOR_NAMES = {"R": "red", "G": "green", "B": "blue"}


@dataclass
class Problem:
    n: int
    edges: tuple[tuple[int, int], ...]   # sorted (u<v) pairs
    has_solution: bool = True
    one_solution: Optional[tuple[str, ...]] = None  # color per vertex if solvable

    def adj(self) -> dict[int, set[int]]:
        a: dict[int, set[int]] = {i: set() for i in range(self.n)}
        for u, v in self.edges:
            a[u].add(v); a[v].add(u)
        return a


# ---------- problem generation ----------


def _is_3_colorable(n: int, edges: tuple[tuple[int, int], ...]
                     ) -> Optional[tuple[str, ...]]:
    """Backtracking 3-coloring search. Returns the first valid coloring or None."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    color = [None] * n

    def go(i: int) -> bool:
        if i == n:
            return True
        for c in COLORS:
            ok = all(color[j] != c for j in adj[i])
            if ok:
                color[i] = c
                if go(i + 1):
                    return True
                color[i] = None
        return False

    if go(0):
        return tuple(color)  # type: ignore
    return None


def generate_problem(n: int, density: float, rng: random.Random,
                      max_attempts: int = 50) -> Problem:
    """Generate a random 3-colorable graph with n vertices and approx
    `density * n*(n-1)/2` edges. Retries if non-colorable.
    """
    for _ in range(max_attempts):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < density:
                    edges.append((u, v))
        edges_t = tuple(sorted(edges))
        sol = _is_3_colorable(n, edges_t)
        if sol is not None:
            return Problem(n=n, edges=edges_t, has_solution=True,
                            one_solution=sol)
    # Fallback: empty graph (always colorable)
    return Problem(n=n, edges=tuple(), has_solution=True,
                    one_solution=tuple([COLORS[0]] * n))


# ---------- tree enumeration ----------


@dataclass
class Node:
    node_id: int
    state: tuple[tuple[int, str], ...]   # ((v_id, color), ...) sorted by v_id
    parent: Optional[int]
    color_used: Optional[str]
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_complete: bool = False             # all vertices colored
    v_value: int = -1                     # remaining-vertices distance to a complete state


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


def _next_uncolored(state: tuple[tuple[int, str], ...], n: int) -> Optional[int]:
    colored = {v for v, _ in state}
    for i in range(n):
        if i not in colored:
            return i
    return None


def _conflicts(state: tuple[tuple[int, str], ...], v: int, c: str,
                adj: dict[int, set[int]]) -> bool:
    state_dict = dict(state)
    for nb in adj[v]:
        if nb in state_dict and state_dict[nb] == c:
            return True
    return False


def enumerate_tree(problem: Problem, max_nodes: int = 5000) -> Tree:
    nodes: list[Node] = []
    seen: dict = {}
    root = Node(node_id=0, state=tuple(), parent=None, color_used=None,
                 depth=0, is_complete=False)
    nodes.append(root); seen[root.state] = 0
    adj = problem.adj()

    frontier = [0]
    while frontier and len(nodes) < max_nodes:
        next_frontier = []
        for pid in frontier:
            parent = nodes[pid]
            v = _next_uncolored(parent.state, problem.n)
            if v is None:
                # all colored — leaf
                continue
            for c in COLORS:
                if _conflicts(parent.state, v, c, adj):
                    continue
                new_state = tuple(sorted(parent.state + ((v, c),),
                                            key=lambda x: x[0]))
                if new_state in seen:
                    nid = seen[new_state]
                    if nid not in parent.children:
                        parent.children.append(nid)
                    continue
                nid = len(nodes)
                child = Node(node_id=nid, state=new_state, parent=pid,
                              color_used=c, depth=parent.depth + 1,
                              is_complete=(len(new_state) == problem.n))
                nodes.append(child); seen[new_state] = nid
                parent.children.append(nid)
                if not child.is_complete:
                    next_frontier.append(nid)
                if len(nodes) >= max_nodes:
                    break
            if len(nodes) >= max_nodes:
                break
        frontier = next_frontier

    # v-values: BFS from complete leaves
    complete_ids = [n.node_id for n in nodes if n.is_complete]
    if complete_ids:
        adj_tree: dict[int, list[int]] = {n.node_id: [] for n in nodes}
        for n in nodes:
            if n.parent is not None:
                adj_tree[n.parent].append(n.node_id)
                adj_tree[n.node_id].append(n.parent)
        from collections import deque
        dist: dict[int, int] = {nid: 0 for nid in complete_ids}
        q = deque(complete_ids)
        while q:
            cur = q.popleft()
            for nb in adj_tree[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        for n in nodes:
            n.v_value = dist.get(n.node_id, -1)
    return Tree(problem=problem, nodes=nodes)


# ---------- rendering ----------


def render_state(problem: Problem,
                  state: tuple[tuple[int, str], ...]) -> str:
    parts = []
    parts.append(f"Graph: {problem.n} vertices "
                  f"({', '.join(f'V{i}' for i in range(problem.n))})")
    parts.append(f"Edges: {', '.join(f'(V{u},V{v})' for u, v in problem.edges)}")
    if state:
        parts.append("Colored so far:")
        for v, c in state:
            parts.append(f"  V{v} = {COLOR_NAMES[c]}")
    uncolored = sorted({i for i in range(problem.n)} - {v for v, _ in state})
    parts.append(f"Uncolored: {', '.join(f'V{i}' for i in uncolored) or '(none)'}")
    return "\n".join(parts)


def format_question(problem: Problem) -> str:
    """Eval prompt: tells the model what to output."""
    edges_str = ", ".join(f"(V{u},V{v})" for u, v in problem.edges)
    if not edges_str:
        edges_str = "(no edges)"
    return (f"Graph 3-coloring task.\n"
            f"Vertices: {', '.join(f'V{i}' for i in range(problem.n))}\n"
            f"Edges: {edges_str}\n"
            f"Assign each vertex one color from {{R, G, B}} such that "
            f"adjacent vertices have different colors. "
            f"Output one assignment per line in the form 'V<i> = <color>'.")


def format_gold_trajectory(problem: Problem,
                            coloring: tuple[str, ...],
                            with_planning_tokens: bool = False) -> str:
    """Render a step-by-step gold trajectory (one line per vertex assignment)."""
    lines = []
    for i, c in enumerate(coloring):
        prefix = f"<PLAN:{c}> " if with_planning_tokens else ""
        lines.append(f"{prefix}V{i} = {COLOR_NAMES[c]}")
    if with_planning_tokens:
        lines.append("<PLAN:ANS> Done.")
    else:
        lines.append("Done.")
    return "\n".join(lines)


# ---------- scoring ----------


_ASSIGN_RE = re.compile(
    r"V(\d+)\s*[:=]\s*(red|green|blue|R|G|B)\b", re.I)


def parse_coloring(generation: str, problem: Problem) -> dict[int, str]:
    """Extract a {vertex_id: color_letter} mapping from the model output."""
    out: dict[int, str] = {}
    for m in _ASSIGN_RE.finditer(generation):
        v = int(m.group(1))
        if v < 0 or v >= problem.n:
            continue
        c = m.group(2).upper()[0]
        if c not in COLORS:
            continue
        if v in out:   # take FIRST assignment per vertex
            continue
        out[v] = c
    return out


def score_coloring(problem: Problem, coloring: dict[int, str]) -> bool:
    """All vertices colored AND every edge satisfied."""
    if len(coloring) != problem.n:
        return False
    for u, v in problem.edges:
        if coloring[u] == coloring[v]:
            return False
    return True


if __name__ == "__main__":
    rng = random.Random(1234)
    for n in [5, 6]:
        for d in [0.2, 0.4, 0.6]:
            p = generate_problem(n, d, rng)
            t = enumerate_tree(p)
            n_complete = sum(1 for nd in t.nodes if nd.is_complete)
            print(f"n={n} d={d}: {len(p.edges)} edges, "
                   f"tree {len(t.nodes)} nodes, {n_complete} complete leaves, "
                   f"max v-value = {max((nd.v_value for nd in t.nodes), default=-1)}")
            print(f"  one solution: {p.one_solution}")
            print(f"  format_question: {format_question(p)[:100]}")
