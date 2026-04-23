"""Varied-target Game-of-24 problem generation.

Reuses the tree enumeration from [src/tree_data.py](src/tree_data.py) to
extract every (pool, target) pair reachable through integer-only intermediate
states. The pool is the `remaining` tuple at some internal node; the target is
the value of some terminal descendant of that node. All values must be
positive integers — fractional intermediate states are skipped so the problem
stays in the integer arithmetic regime our parser and oracle handle.

Each returned pair comes with the minimum number of ops needed (`n_steps`),
computed as the depth gap between the sub-root and the terminal descendant.

Note: the same (sorted pool, target) key can appear via different parent
problems. Dedup at the caller's level if cross-problem uniqueness matters.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterator

from src.tree_data import enumerate_tree


def _is_positive_int_frac(f: Fraction) -> bool:
    return f.denominator == 1 and f.numerator > 0


def iter_varied_pairs(problem: str) -> Iterator[dict]:
    """Yield every valid (pool, target) pair reachable from `problem`.

    A pair is valid iff:
      - The sub-root's `remaining` is all positive integers (no fractions).
      - The terminal descendant's single value is a positive integer.
      - `n_steps == len(pool) - 1` (always true for Game-of-24 topology).
    """
    tree = enumerate_tree(problem)

    for i, node in enumerate(tree.nodes):
        if node.is_terminal:
            continue
        if not all(_is_positive_int_frac(x) for x in node.remaining):
            continue

        pool = [int(x) for x in node.remaining]
        sub_root_depth = node.depth

        # BFS over descendants to find every terminal reachable from this node.
        stack: list[int] = list(node.children)
        visited: set = {i}
        while stack:
            j = stack.pop()
            if j in visited:
                continue
            visited.add(j)
            child = tree.nodes[j]
            if child.is_terminal:
                val = child.remaining[0]
                if not _is_positive_int_frac(val):
                    continue
                yield {
                    "pool": sorted(pool),
                    "target": int(val),
                    "n_steps": child.depth - sub_root_depth,
                    "source_problem": problem,
                }
            else:
                stack.extend(child.children)


def collect_unique_pairs(problem: str) -> list[dict]:
    """Dedup within one tree by (sorted pool, target).

    Keeps the minimum-n_steps representative when duplicates exist (the tree
    may contain multiple paths to the same (pool, target) pair via different
    sub-roots that happen to share `remaining`, or via different parent ops).
    """
    best: dict = {}
    for pair in iter_varied_pairs(problem):
        key = (tuple(pair["pool"]), pair["target"])
        if key not in best or pair["n_steps"] < best[key]["n_steps"]:
            best[key] = pair
    return list(best.values())
