"""Oracle for Countdown DAgger training.

Generalizes oracle_24 to variable target and variable pool size. Uses standard
Countdown rules (integer arithmetic):
- subtraction must be non-negative (a - b valid iff a >= b)
- division must be exact (a / b valid iff b != 0 and a % b == 0)

Because the target varies per problem, we use a per-problem cache rather than
a global lru_cache. One CountdownOracle instance = one problem's memo table.
"""
from __future__ import annotations

from typing import Iterable


# (op_symbol, fn, commutative). fn returns None if the op is illegal at (a, b).
OPS = [
    ("+", lambda a, b: a + b, True),
    ("-", lambda a, b: a - b if a >= b else None, False),
    ("*", lambda a, b: a * b, True),
    ("/", lambda a, b: a // b if b != 0 and a % b == 0 else None, False),
]


def _apply(op_sym: str, a: int, b: int):
    for sym, fn, _ in OPS:
        if sym == op_sym:
            return fn(a, b)
    return None


class CountdownOracle:
    """Per-problem oracle with a private memo cache.

    Cache key = sorted pool tuple (target is fixed for this oracle). Intended
    usage: one instance per (pool, target) problem, reused across all
    step-boundary queries within that problem's rollout.
    """

    __slots__ = ("target", "_cache")

    def __init__(self, target: int):
        self.target = target
        self._cache: dict[tuple[int, ...], bool] = {}

    def can_reach(self, remaining: tuple[int, ...]) -> bool:
        """True iff some sequence of legal ops on `remaining` ends in target.

        `remaining` MUST be a sorted tuple. Early-terminates on first success.
        """
        if len(remaining) == 1:
            return remaining[0] == self.target
        cached = self._cache.get(remaining)
        if cached is not None:
            return cached

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
                        continue  # dedup commutative ordering
                    r = fn(a, b)
                    if r is None:
                        continue
                    new_state = tuple(sorted(rest + (r,)))
                    dedup_key = (sym, a, b, new_state)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    if self.can_reach(new_state):
                        self._cache[remaining] = True
                        return True
        self._cache[remaining] = False
        return False

    def winning_ops(self, remaining: Iterable[int]
                    ) -> list[tuple[str, int, int, int]]:
        """Return all winning (op_sym, a, b, r) applicable to `remaining`.

        Mirrors oracle_24.winning_ops: for commutative ops both orderings
        (a, b) and (b, a) are included when a != b; for non-commutative ops
        only the legal ordering is emitted.
        """
        rem = tuple(sorted(remaining))
        if len(rem) < 2:
            return []
        winners: list[tuple[str, int, int, int]] = []
        seen_keys: set = set()
        n = len(rem)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = rem[i], rem[j]
                rest = tuple(rem[k] for k in range(n) if k != i and k != j)
                for sym, fn, _comm in OPS:
                    r = fn(a, b)
                    if r is None:
                        continue
                    new_state = tuple(sorted(rest + (r,)))
                    if not self.can_reach(new_state):
                        continue
                    key = (sym, a, b, r)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    winners.append((sym, a, b, r))
        return winners


def validate_step(remaining: Iterable[int], a: int, op: str, b: int, r: int
                  ) -> tuple[bool, str]:
    """Check whether "a op b = r" is a legal step from `remaining`."""
    rem = list(sorted(remaining))
    try:
        rem.remove(a)
    except ValueError:
        return False, f"operand {a} not in remaining {rem}"
    try:
        rem.remove(b)
    except ValueError:
        return False, f"operand {b} not in remaining {rem + [a]}"
    computed = _apply(op, a, b)
    if computed is None:
        return False, f"op {op} illegal at ({a}, {b}) under Countdown rules"
    if computed != r:
        return False, f"arithmetic mismatch: {a} {op} {b} = {computed} != {r}"
    return True, ""


def apply_step(remaining: Iterable[int], a: int, b: int, r: int) -> tuple:
    """Return the new sorted remaining after one step. Assumes validated."""
    rem = list(sorted(remaining))
    rem.remove(a)
    rem.remove(b)
    rem.append(r)
    return tuple(sorted(rem))
