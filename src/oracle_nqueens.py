"""N-Queens oracle: problem generation, tree enumeration, scoring.

Canonical 1-indexed text format (parallels Game-of-24 trajectory style).
State = tuple of column placements for rows 1..k (1-indexed columns).
Action at depth k = column to place at row k+1 (1..N), conflict-free.
Tree branches over column choice at each step. Terminal = depth N.

Public API:
- Problem(N, prefix)
- generate_problem(N, n_pre_placed, rng) -> Problem  (samples valid prefix)
- enumerate_tree(problem, max_nodes) -> Tree
- render_state(problem, state) -> str (text input for Stage-1 head)
- format_question(problem) -> str  (eval prompt header)
- format_step_text(state_before, action) -> str  (one Step line)
- format_gold_trajectory(N, solution) -> str  (full trajectory)
- parse_step(step_body, problem, state) -> int | None  (extracts column)
- parse_solution(generation) -> list[int] | None
- score_solution(N, solution) -> bool
- solve_lex_min(N, prefix=None) -> list[int] | None
- available_columns(N, placed, next_row) -> list[int]
- all_solutions(N) -> list[list[int]]
- all_distinct_prefixes(N, k) -> list[list[int]]
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


def _conflicts(placed: list[tuple[int, int]], r: int, c: int) -> bool:
    """True if a queen at (r,c) attacks any queen in `placed` (1-indexed)."""
    for (rp, cp) in placed:
        if cp == c:
            return True
        if abs(rp - r) == abs(cp - c):
            return True
    return False


def available_columns(N: int, placed: list[tuple[int, int]],
                      next_row: int) -> list[int]:
    """Return sorted list of 1-indexed columns where a queen can be safely
    placed at row `next_row` given prior `placed` queens."""
    return [c for c in range(1, N + 1)
            if not _conflicts(placed, next_row, c)]


def solve_lex_min(N: int, prefix: Optional[list[tuple[int, int]]] = None
                  ) -> Optional[list[int]]:
    """Return the lex-smallest solution (column-per-row, 1-indexed) extending
    `prefix`, or None if unsolvable."""
    placed = list(prefix or [])
    next_row = len(placed) + 1
    if next_row > N:
        return [c for (_, c) in placed]
    for c in range(1, N + 1):
        if _conflicts(placed, next_row, c):
            continue
        sol = solve_lex_min(N, placed + [(next_row, c)])
        if sol is not None:
            return sol
    return None


def format_gold_trajectory(N: int, solution: list[int]) -> str:
    """Render the canonical step-by-step trajectory text for `solution`."""
    lines = [f"Board size: {N}"]
    placed: list[tuple[int, int]] = []
    for k, c in enumerate(solution, 1):
        placed.append((k, c))
        lines.append(f"Step {k}: Place queen in row {k} at column {c}.")
        placed_str = ",".join(f"({r},{cc})" for r, cc in placed)
        lines.append(f"  Placed: [{placed_str}]")
        if k < N:
            avail = available_columns(N, placed, k + 1)
            avail_str = ", ".join(str(x) for x in avail)
            lines.append(f"  Available for row {k+1}: [{avail_str}]")
    sol_str = ", ".join(str(c) for c in solution)
    lines.append(f"Solution: [{sol_str}]")
    return "\n".join(lines)


def parse_solution(generation: str) -> Optional[list[int]]:
    """Extract the column list from the final 'Solution: [...]' line.

    Tolerates: 'Solution: [1, 3, 5, 2, 4]', 'Solution: 1 3 5 2 4',
    'Solution: 1, 3, 5, 2, 4'.
    """
    bracket = re.search(r"Solution\s*[:=]?\s*\[([^\]]*)\]", generation)
    if bracket:
        body = bracket.group(1)
    else:
        bare = re.search(r"Solution\s*[:=]?\s*([\d\s,]+)", generation)
        if not bare:
            return None
        body = bare.group(1)
    parts = re.split(r"[,\s]+", body.strip())
    try:
        return [int(p) for p in parts if p]
    except ValueError:
        return None


def all_solutions(N: int) -> list[list[int]]:
    """Return every valid N-Queens solution as a 1-indexed column list."""
    sols: list[list[int]] = []

    def _search(placed: list[tuple[int, int]], next_row: int) -> None:
        if next_row > N:
            sols.append([c for _, c in placed])
            return
        for c in range(1, N + 1):
            if _conflicts(placed, next_row, c):
                continue
            _search(placed + [(next_row, c)], next_row + 1)

    _search([], 1)
    return sols


def all_distinct_prefixes(N: int, k: int) -> list[list[int]]:
    """All distinct length-k 1-indexed column prefixes (rows 1..k) that
    extend to a valid full solution at board size N."""
    if k == 0:
        return [[]]
    seen: set[tuple[int, ...]] = set()
    for s in all_solutions(N):
        seen.add(tuple(s[:k]))
    return [list(p) for p in sorted(seen)]


def render_prefix_steps(N: int, prefix_cols: list[int]) -> str:
    """Render step lines for an already-placed prefix (no Solution: line).

    Returns trailing-newline-free text. Used to build prompts where the
    model continues from row len(prefix_cols)+1.
    """
    lines: list[str] = []
    placed: list[tuple[int, int]] = []
    for r, c in enumerate(prefix_cols, 1):
        placed.append((r, c))
        lines.append(f"Step {r}: Place queen in row {r} at column {c}.")
        placed_str = ",".join(f"({rr},{cc})" for rr, cc in placed)
        lines.append(f"  Placed: [{placed_str}]")
        if r < N:
            avail = available_columns(N, placed, r + 1)
            avail_str = ", ".join(str(x) for x in avail)
            lines.append(f"  Available for row {r+1}: [{avail_str}]")
    return "\n".join(lines)


def score_solution(N: int, solution: Optional[list[int]]) -> bool:
    """Validate a candidate N-Queens solution. 1-indexed columns."""
    if solution is None or len(solution) != N:
        return False
    if any(c < 1 or c > N for c in solution):
        return False
    placed = [(r, c) for r, c in enumerate(solution, 1)]
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            ri, ci = placed[i]
            rj, cj = placed[j]
            if ci == cj or abs(ri - rj) == abs(ci - cj):
                return False
    return True


# ---------- Problem + Tree (for HypPlan training) ----------

@dataclass
class Problem:
    """N-Queens problem with optional pre-placed queens.

    `prefix` is a length-k 1-indexed column tuple for rows 1..k.
    The model must extend the prefix to a full valid placement.
    """
    N: int
    prefix: tuple[int, ...] = ()

    def initial_state(self) -> tuple[int, ...]:
        return tuple(self.prefix)


@dataclass
class Node:
    node_id: int
    state: tuple[int, ...]   # column placements rows 1..len(state)
    parent: Optional[int]
    action: Optional[int]    # column placed at this row (None for root)
    children: list[int] = field(default_factory=list)
    depth: int = 0
    is_solved: bool = False  # all N queens placed (terminal success)
    v_value: int = -1         # distance to nearest solved descendant


@dataclass
class Tree:
    problem: Problem
    nodes: list[Node]


def _has_extension(N: int, state: tuple[int, ...]) -> bool:
    """True iff state extends to at least one valid full placement."""
    placed = [(r, c) for r, c in enumerate(state, 1)]
    return solve_lex_min(N, placed) is not None


def enumerate_tree(problem: Problem, max_nodes: int = 5000) -> Tree:
    """BFS-enumerate all valid extensions of `problem.prefix`. Pruned to
    branches that lead to at least one full solution."""
    nodes: list[Node] = []
    seen: dict[tuple[int, ...], int] = {}
    root = Node(node_id=0, state=problem.initial_state(), parent=None,
                action=None, depth=len(problem.prefix),
                is_solved=(len(problem.prefix) == problem.N))
    nodes.append(root)
    seen[root.state] = 0

    frontier = [0]
    while frontier and len(nodes) < max_nodes:
        next_frontier: list[int] = []
        for pid in frontier:
            parent = nodes[pid]
            if parent.is_solved:
                continue
            next_row = len(parent.state) + 1
            placed = [(r, c) for r, c in enumerate(parent.state, 1)]
            for c in range(1, problem.N + 1):
                if _conflicts(placed, next_row, c):
                    continue
                new_state = parent.state + (c,)
                if new_state in seen:
                    nid = seen[new_state]
                    if nid not in parent.children:
                        parent.children.append(nid)
                    continue
                # Prune branches that can't reach a full placement.
                if not _has_extension(problem.N, new_state):
                    continue
                nid = len(nodes)
                child = Node(
                    node_id=nid, state=new_state, parent=pid,
                    action=c, depth=parent.depth + 1,
                    is_solved=(len(new_state) == problem.N),
                )
                nodes.append(child)
                seen[new_state] = nid
                parent.children.append(nid)
                if not child.is_solved:
                    next_frontier.append(nid)
                if len(nodes) >= max_nodes:
                    break
            if len(nodes) >= max_nodes:
                break
        frontier = next_frontier

    # v-values: BFS from solved leaves outward.
    solved_ids = [n.node_id for n in nodes if n.is_solved]
    if solved_ids:
        adj: dict[int, list[int]] = {n.node_id: [] for n in nodes}
        for n in nodes:
            if n.parent is not None:
                adj[n.parent].append(n.node_id)
                adj[n.node_id].append(n.parent)
        dist: dict[int, int] = {nid: 0 for nid in solved_ids}
        q: deque[int] = deque(solved_ids)
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        for n in nodes:
            n.v_value = dist.get(n.node_id, -1)
    return Tree(problem=problem, nodes=nodes)


def render_state(problem: Problem, state: tuple[int, ...]) -> str:
    """Plain-text rendering of a partial board state for the head's input."""
    parts = [f"Board size: {problem.N}"]
    if state:
        placed_str = ",".join(f"({r},{c})" for r, c in
                              enumerate(state, 1))
        parts.append(f"Placed: [{placed_str}]")
    else:
        parts.append("Placed: []")
    next_row = len(state) + 1
    if next_row <= problem.N:
        placed = [(r, c) for r, c in enumerate(state, 1)]
        avail = [c for c in range(1, problem.N + 1)
                 if not _conflicts(placed, next_row, c)]
        parts.append(f"Next row: {next_row}")
        parts.append(f"Available columns: "
                     f"{', '.join(str(c) for c in avail) or '(none)'}")
    else:
        parts.append("All queens placed.")
    return "\n".join(parts)


def format_question(problem: Problem) -> str:
    """Eval prompt header for the model."""
    if problem.prefix:
        prefix_lines = []
        for r, c in enumerate(problem.prefix, 1):
            prefix_lines.append(
                f"  row {r} col {c}")
        pre = ("\nThe following queens are already placed:\n"
               + "\n".join(prefix_lines))
    else:
        pre = ""
    return (f"N-Queens task. Place {problem.N} queens on an "
            f"{problem.N}x{problem.N} board, one per row, so that no "
            f"two queens share the same column or diagonal.{pre}\n"
            f"At each step, place a queen in the next empty row at a "
            f"valid column. Output one Step line per row, then the "
            f"final 'Solution: [c1, c2, ..., c{problem.N}]' line "
            f"with column placements (1-indexed).")


def format_step_text(state_before: tuple[int, ...], action: int) -> str:
    """Render one Step line: 'Step k: Place queen in row k at column c.'"""
    next_row = len(state_before) + 1
    return f"Step {next_row}: Place queen in row {next_row} at column {action}."


_STEP_RE = re.compile(
    r"row\s+(\d+)\s+at\s+column\s+(\d+)|col\s*[:=]?\s*(\d+)|"
    r"column\s+(\d+)", re.IGNORECASE)


def parse_step(step_body: str, problem: Problem,
               state: tuple[int, ...]) -> Optional[int]:
    """Parse a Step body and return the column placed (1-indexed), or None.

    Accepts: 'Place queen in row R at column C.', 'col C', 'column C', etc.
    """
    next_row = len(state) + 1
    m = _STEP_RE.search(step_body)
    if not m:
        # Bare integer fallback
        m2 = re.search(r"\b(\d+)\b", step_body)
        if not m2:
            return None
        c = int(m2.group(1))
    else:
        # Try the matched group(s) for the column number
        c = None
        if m.group(1) is not None and m.group(2) is not None:
            r = int(m.group(1))
            if r != next_row:
                return None
            c = int(m.group(2))
        else:
            c = int(m.group(3) or m.group(4))
    if c is None or c < 1 or c > problem.N:
        return None
    placed = [(r, ci) for r, ci in enumerate(state, 1)]
    if _conflicts(placed, next_row, c):
        return None
    return c


def is_solved(state: tuple[int, ...], problem: Problem) -> bool:
    """Terminal predicate: all N rows placed (and by construction valid)."""
    return len(state) == problem.N


def winning_steps(state: tuple[int, ...],
                  problem: Problem) -> list[tuple[int, tuple[int, ...]]]:
    """Return list of (column, new_state) pairs that extend `state` to a
    solution-reachable subtree."""
    if len(state) >= problem.N:
        return []
    next_row = len(state) + 1
    placed = [(r, c) for r, c in enumerate(state, 1)]
    out: list[tuple[int, tuple[int, ...]]] = []
    for c in range(1, problem.N + 1):
        if _conflicts(placed, next_row, c):
            continue
        new_state = state + (c,)
        if _has_extension(problem.N, new_state):
            out.append((c, new_state))
    return out


def validate_step(state: tuple[int, ...], action: int,
                  problem: Problem) -> tuple[bool, tuple[int, ...]]:
    """Verify `action` is a winning column at the next row of `state`."""
    next_row = len(state) + 1
    if action < 1 or action > problem.N or next_row > problem.N:
        return False, state
    placed = [(r, c) for r, c in enumerate(state, 1)]
    if _conflicts(placed, next_row, action):
        return False, state
    new_state = state + (action,)
    if not _has_extension(problem.N, new_state):
        return False, state
    return True, new_state


def generate_problem(N: int, n_pre_placed: int,
                     rng: random.Random) -> Problem:
    """Sample a valid (N, k=n_pre_placed)-prefix problem. The prefix is
    drawn uniformly from the universe of length-k prefixes that extend
    to a valid full solution."""
    if n_pre_placed == 0:
        return Problem(N=N, prefix=())
    prefs = all_distinct_prefixes(N, n_pre_placed)
    chosen = rng.choice(prefs)
    return Problem(N=N, prefix=tuple(chosen))
