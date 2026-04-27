"""Oracle for single-variable linear-equation solving (Group A OOD #1).

Equations are of the form
   a1·x + a2·x + ... + b1 + b2 + ... = c1·x + c2·x + ... + d1 + d2 + ...
with integer coefficients and integer solution x* ∈ [-9, 9] \\ {0}.

The state on each side is a multiset of x-coefficients and a multiset of
integer constants. Solving proceeds in a canonical 5-op order:
  1. combine_lhs_x      — sum LHS x-coefficients into one term
  2. combine_lhs_const  — sum LHS constants into one
  3. combine_rhs_x      — sum RHS x-coefficients into one
  4. combine_rhs_const  — sum RHS constants into one
  5. move_x_to_lhs      — subtract RHS x-coef from both sides
  6. move_const_to_rhs  — subtract LHS const from both sides
  7. divide_both        — divide both sides by LHS x-coef (must be integer)

A "solving step" in our difficulty count is any of those that's actually
needed (e.g., LHS has only one x-term ⇒ combine_lhs_x is a no-op and not
counted).

Difficulty knob `k`:
- k=3: minimum nontrivial — single x-term on each side, single const each
       side, must move_x + move_const + divide.
- k=4: one extra combine — adds a multi-term x-coefficient on one side.
- k=5: two extra combines — multi-term x AND multi-term const, or both
       sides have multi-term x.

Mirrors the Tree/Node interface of `oracle_pronto`/`oracle_blocksworld` so
it plugs into the existing eval/training pipeline with minimal glue.
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional


# ---------------------------- Data classes ----------------------------

@dataclass(frozen=True)
class State:
    """One side's contents are a tuple of ints (sorted for canonicality)."""
    lhs_x: tuple[int, ...]      # coefficients of x on LHS, sorted
    lhs_c: tuple[int, ...]      # constants on LHS, sorted
    rhs_x: tuple[int, ...]
    rhs_c: tuple[int, ...]

    def render(self) -> str:
        def side(xs: tuple[int, ...], cs: tuple[int, ...]) -> str:
            parts: list[str] = []
            for c in xs:
                if c == 0:
                    continue
                if c == 1:
                    parts.append("x")
                elif c == -1:
                    parts.append("-x")
                else:
                    parts.append(f"{c}*x")
            for v in cs:
                parts.append(f"{v}")
            if not parts:
                return "0"
            # Render as "a + b - c + d" style.
            out = parts[0]
            for p in parts[1:]:
                if p.startswith("-"):
                    out += f" - {p[1:]}"
                else:
                    out += f" + {p}"
            return out
        return f"{side(self.lhs_x, self.lhs_c)} = {side(self.rhs_x, self.rhs_c)}"


@dataclass(frozen=True)
class Op:
    kind: str               # 'combine_lhs_x' / 'combine_lhs_const' / ...
    arg: Optional[int] = None

    def render(self) -> str:
        if self.kind == "combine_lhs_x":
            return "combine like x-terms on the left"
        if self.kind == "combine_lhs_const":
            return "combine constants on the left"
        if self.kind == "combine_rhs_x":
            return "combine like x-terms on the right"
        if self.kind == "combine_rhs_const":
            return "combine constants on the right"
        if self.kind == "move_x_to_lhs":
            return f"subtract {self.arg}*x from both sides"
        if self.kind == "move_const_to_rhs":
            return f"subtract {self.arg} from both sides"
        if self.kind == "divide_both":
            return f"divide both sides by {self.arg}"
        return f"<unknown op {self.kind}>"


@dataclass
class Problem:
    initial: State
    solution: int
    raw: str = ""

    def render_problem(self) -> str:
        return (
            f"Solve for x:\n"
            f"  {self.initial.render()}"
        )


@dataclass
class Node:
    node_id: int
    state: State
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


# ---------------------------- State predicates ----------------------------

def _sumtup(t: tuple[int, ...]) -> int:
    return sum(t)


def is_combined_lhs_x(s: State) -> bool:
    return len(s.lhs_x) <= 1


def is_combined_lhs_c(s: State) -> bool:
    return len(s.lhs_c) <= 1


def is_combined_rhs_x(s: State) -> bool:
    return len(s.rhs_x) <= 1


def is_combined_rhs_c(s: State) -> bool:
    return len(s.rhs_c) <= 1


def is_solved(s: State, x_star: int) -> bool:
    """Solved form: x = K for the integer solution K."""
    if s.lhs_x != (1,) or s.lhs_c not in ((), (0,)):
        return False
    if s.rhs_x not in ((), (0,)):
        return False
    if s.rhs_c == (x_star,):
        return True
    if s.rhs_c == () and x_star == 0:
        return True
    return False


def _canon(t: tuple[int, ...]) -> tuple[int, ...]:
    """Sort and drop trailing zero singletons (we keep an empty tuple for
    'no constant' rather than a (0,) singleton)."""
    cleaned = tuple(sorted(t))
    if cleaned == (0,):
        return ()
    return cleaned


# ---------------------------- Op application ----------------------------

def apply_op(s: State, op: Op) -> Optional[State]:
    """Return a new State, or None if op is illegal in `s`."""
    if op.kind == "combine_lhs_x":
        if len(s.lhs_x) <= 1:
            return None
        return State(
            lhs_x=_canon((sum(s.lhs_x),)),
            lhs_c=s.lhs_c,
            rhs_x=s.rhs_x,
            rhs_c=s.rhs_c,
        )
    if op.kind == "combine_lhs_const":
        if len(s.lhs_c) <= 1:
            return None
        return State(
            lhs_x=s.lhs_x,
            lhs_c=_canon((sum(s.lhs_c),)),
            rhs_x=s.rhs_x,
            rhs_c=s.rhs_c,
        )
    if op.kind == "combine_rhs_x":
        if len(s.rhs_x) <= 1:
            return None
        return State(
            lhs_x=s.lhs_x, lhs_c=s.lhs_c,
            rhs_x=_canon((sum(s.rhs_x),)),
            rhs_c=s.rhs_c,
        )
    if op.kind == "combine_rhs_const":
        if len(s.rhs_c) <= 1:
            return None
        return State(
            lhs_x=s.lhs_x, lhs_c=s.lhs_c,
            rhs_x=s.rhs_x,
            rhs_c=_canon((sum(s.rhs_c),)),
        )
    if op.kind == "move_x_to_lhs":
        # Allowed only when both sides have a single x-term (combined).
        if not (is_combined_lhs_x(s) and is_combined_rhs_x(s)):
            return None
        rhs_x_val = s.rhs_x[0] if s.rhs_x else 0
        if rhs_x_val == 0:
            return None  # nothing to move
        if op.arg != rhs_x_val:
            return None
        new_lhs_x = (s.lhs_x[0] if s.lhs_x else 0) - rhs_x_val
        return State(
            lhs_x=_canon((new_lhs_x,)),
            lhs_c=s.lhs_c,
            rhs_x=(),
            rhs_c=s.rhs_c,
        )
    if op.kind == "move_const_to_rhs":
        if not (is_combined_lhs_c(s) and is_combined_rhs_c(s)):
            return None
        lhs_c_val = s.lhs_c[0] if s.lhs_c else 0
        if lhs_c_val == 0:
            return None
        if op.arg != lhs_c_val:
            return None
        new_rhs_c = (s.rhs_c[0] if s.rhs_c else 0) - lhs_c_val
        return State(
            lhs_x=s.lhs_x,
            lhs_c=(),
            rhs_x=s.rhs_x,
            rhs_c=_canon((new_rhs_c,)),
        )
    if op.kind == "divide_both":
        # Only legal when LHS = (c,)*x, RHS = (k,) const, and c | k.
        if (s.lhs_x in ((), (0,))
            or len(s.lhs_x) != 1
            or s.lhs_c not in ((), (0,))
            or s.rhs_x not in ((), (0,))
            or len(s.rhs_c) != 1
        ):
            return None
        c = s.lhs_x[0]
        k = s.rhs_c[0]
        if c == 0:
            return None
        if k % c != 0:
            return None
        if op.arg != c:
            return None
        return State(
            lhs_x=(1,), lhs_c=(),
            rhs_x=(), rhs_c=_canon((k // c,)),
        )
    return None


def applicable_ops(s: State) -> list[Op]:
    """All ops that are legal in s, in canonical preference order."""
    out: list[Op] = []
    if not is_combined_lhs_x(s):
        out.append(Op("combine_lhs_x"))
    if not is_combined_lhs_c(s):
        out.append(Op("combine_lhs_const"))
    if not is_combined_rhs_x(s):
        out.append(Op("combine_rhs_x"))
    if not is_combined_rhs_c(s):
        out.append(Op("combine_rhs_const"))
    # move_x_to_lhs: both x-sides must be singleton
    if (is_combined_lhs_x(s) and is_combined_rhs_x(s)
        and s.rhs_x and s.rhs_x[0] != 0):
        out.append(Op("move_x_to_lhs", arg=s.rhs_x[0]))
    # move_const_to_rhs
    if (is_combined_lhs_c(s) and is_combined_rhs_c(s)
        and s.lhs_c and s.lhs_c[0] != 0):
        out.append(Op("move_const_to_rhs", arg=s.lhs_c[0]))
    # divide_both: requires fully isolated form
    if (s.lhs_x and len(s.lhs_x) == 1 and s.lhs_x[0] not in (0, 1)
        and s.lhs_c in ((), (0,))
        and s.rhs_x in ((), (0,))
        and s.rhs_c and len(s.rhs_c) == 1
        and s.rhs_c[0] % s.lhs_x[0] == 0):
        out.append(Op("divide_both", arg=s.lhs_x[0]))
    return out


# ---------------------------- Tree enumeration ----------------------------

def enumerate_tree(problem: Problem, max_nodes: int = 1000,
                   max_depth: int = 8) -> Tree:
    root = Node(
        node_id=0, state=problem.initial, parent=None, op_used=None,
        depth=0, is_solved=is_solved(problem.initial, problem.solution),
    )
    nodes: list[Node] = [root]
    seen: dict[State, int] = {problem.initial: 0}
    queue: deque[int] = deque([0])
    while queue and len(nodes) < max_nodes:
        nid = queue.popleft()
        node = nodes[nid]
        if node.is_solved or node.depth >= max_depth:
            continue
        for op in applicable_ops(node.state):
            ns = apply_op(node.state, op)
            if ns is None:
                continue
            if ns in seen:
                cid = seen[ns]
            else:
                cid = len(nodes)
                nodes.append(Node(
                    node_id=cid, state=ns, parent=nid, op_used=op,
                    depth=node.depth + 1,
                    is_solved=is_solved(ns, problem.solution),
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


# ---------------------------- Lightweight oracle ----------------------------

def winning_steps(s: State, problem: Problem,
                  max_nodes: int = 500) -> list[Op]:
    """Ops in `s` that strictly decrease BFS distance to a solved state."""
    if is_solved(s, problem.solution):
        return []

    def min_dist(start: State) -> Optional[int]:
        if is_solved(start, problem.solution):
            return 0
        seen: dict[State, int] = {start: 0}
        q: deque[State] = deque([start])
        while q and len(seen) < max_nodes:
            cur = q.popleft()
            d = seen[cur]
            for op in applicable_ops(cur):
                ns = apply_op(cur, op)
                if ns is None or ns in seen:
                    continue
                nd = d + 1
                if is_solved(ns, problem.solution):
                    return nd
                seen[ns] = nd
                q.append(ns)
        return None

    s_dist = min_dist(s)
    if s_dist is None:
        return []
    out: list[Op] = []
    for op in applicable_ops(s):
        ns = apply_op(s, op)
        if ns is None:
            continue
        nd = min_dist(ns)
        if nd is not None and nd == s_dist - 1:
            out.append(op)
    return out


def validate_step(s: State, op: Op) -> tuple[bool, State]:
    ns = apply_op(s, op)
    if ns is None:
        return False, s
    return True, ns


# ---------------------------- Problem generation ----------------------------

def generate_problem(k: int, seed: int = 0,
                     max_attempts: int = 200) -> Problem:
    """Generate a problem solvable in EXACTLY `k` canonical steps.

    Strategy: pick x*, build the appropriate state shape for difficulty k,
    pick coefficients, then verify the BFS distance to solved is k."""
    rng = random.Random(seed)
    if k < 3:
        raise ValueError("k must be >= 3")

    for _ in range(max_attempts):
        x_star = rng.choice([n for n in range(-9, 10) if n != 0])

        if k == 3:
            # Single x-term and single const on each side.
            # Form: a*x + b = c*x + d with x = (d - b)/(a - c).
            for _ in range(40):
                a = rng.randint(1, 9)
                c = rng.randint(0, 9)
                if a == c:
                    continue
                b = rng.randint(-9, 9)
                d = b + (a - c) * x_star
                if abs(d) > 50:
                    continue
                lhs_x = (a,) if a != 0 else ()
                lhs_c = (b,) if b != 0 else ()
                rhs_x = (c,) if c != 0 else ()
                rhs_c = (d,) if d != 0 else ()
                if not lhs_x or not rhs_x:
                    continue
                if not lhs_c:
                    continue  # no const on LHS would mean k<3
                state = State(lhs_x, lhs_c, rhs_x, rhs_c)
                p = Problem(initial=state, solution=x_star)
                tree = enumerate_tree(p, max_nodes=300, max_depth=k + 2)
                root = tree.nodes[0]
                if root.v_value == k:
                    return p

        elif k == 4:
            # Multi-term x on LHS (a1, a2) plus const, single x on RHS plus const.
            for _ in range(60):
                a1 = rng.randint(1, 5)
                a2 = rng.randint(1, 5)
                a = a1 + a2
                c = rng.randint(0, 5)
                if a == c:
                    continue
                b = rng.randint(-9, 9)
                d = b + (a - c) * x_star
                if abs(d) > 50:
                    continue
                lhs_x = tuple(sorted((a1, a2)))
                lhs_c = (b,) if b != 0 else ()
                rhs_x = (c,) if c != 0 else ()
                rhs_c = (d,) if d != 0 else ()
                if not lhs_c:
                    continue
                state = State(lhs_x, lhs_c, rhs_x, rhs_c)
                p = Problem(initial=state, solution=x_star)
                tree = enumerate_tree(p, max_nodes=400, max_depth=k + 2)
                if tree.nodes[0].v_value == k:
                    return p

        else:  # k == 5
            # Multi-term x AND multi-term const on LHS.
            for _ in range(80):
                a1 = rng.randint(1, 5)
                a2 = rng.randint(1, 5)
                a = a1 + a2
                c = rng.randint(0, 5)
                if a == c:
                    continue
                b1 = rng.randint(-5, 5)
                b2 = rng.randint(-5, 5)
                if b1 == 0 or b2 == 0:
                    continue
                b = b1 + b2
                d = b + (a - c) * x_star
                if abs(d) > 50:
                    continue
                lhs_x = tuple(sorted((a1, a2)))
                lhs_c = tuple(sorted((b1, b2)))
                rhs_x = (c,) if c != 0 else ()
                rhs_c = (d,) if d != 0 else ()
                state = State(lhs_x, lhs_c, rhs_x, rhs_c)
                p = Problem(initial=state, solution=x_star)
                tree = enumerate_tree(p, max_nodes=600, max_depth=k + 2)
                if tree.nodes[0].v_value == k:
                    return p

    raise RuntimeError(
        f"Could not generate linear-equation problem with k={k} after "
        f"{max_attempts} attempts"
    )


# ---------------------------- Rendering ----------------------------

def render_state(problem: Problem, state: State) -> str:
    return (
        f"Solve for x.\n"
        f"Current equation: {state.render()}"
    )


def format_question(problem: Problem) -> str:
    return problem.render_problem()


def format_step_text(op: Op, state_after: State) -> str:
    return f"{op.render()} → {state_after.render()}"


def format_gold_trajectory(problem: Problem, max_steps: int = 10) -> str:
    lines: list[str] = []
    state = problem.initial
    step = 1
    for _ in range(max_steps):
        if is_solved(state, problem.solution):
            break
        wins = winning_steps(state, problem)
        if not wins:
            break
        op = wins[0]
        ns = apply_op(state, op)
        if ns is None:
            break
        lines.append(f"Step {step}: {format_step_text(op, ns)}")
        state = ns
        step += 1
    lines.append(f"Answer: x = {problem.solution}")
    return "\n".join(lines)


# ---------------------------- Step parsing ----------------------------

_STEP_OP_RE = re.compile(
    r"(combine\s+like\s+x-terms\s+on\s+the\s+(left|right)|"
    r"combine\s+constants\s+on\s+the\s+(left|right)|"
    r"subtract\s+(-?\d+)\*x\s+from\s+both\s+sides|"
    r"subtract\s+(-?\d+)\s+from\s+both\s+sides|"
    r"divide\s+both\s+sides\s+by\s+(-?\d+))",
    re.IGNORECASE,
)
_ANSWER_RE = re.compile(r"Answer\s*[:\-]?\s*x\s*=\s*(-?\d+)", re.IGNORECASE)


def parse_step(text: str) -> Optional[Op]:
    """Recognise one canonical op from a model-emitted step body."""
    m = _STEP_OP_RE.search(text)
    if m is None:
        return None
    head = m.group(1).lower()
    if "combine" in head and "x-terms" in head and "left" in head:
        return Op("combine_lhs_x")
    if "combine" in head and "x-terms" in head and "right" in head:
        return Op("combine_rhs_x")
    if "combine" in head and "constants" in head and "left" in head:
        return Op("combine_lhs_const")
    if "combine" in head and "constants" in head and "right" in head:
        return Op("combine_rhs_const")
    if "subtract" in head and "*x" in head:
        return Op("move_x_to_lhs", arg=int(m.group(4)))
    if "subtract" in head:
        return Op("move_const_to_rhs", arg=int(m.group(5)))
    if "divide" in head:
        return Op("divide_both", arg=int(m.group(6)))
    return None


def parse_answer(generation: str) -> Optional[int]:
    m = _ANSWER_RE.search(generation)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def score_answer(prediction: Optional[int], gold: int) -> bool:
    return prediction is not None and prediction == gold
