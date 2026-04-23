"""Oracle for varied-target Game-of-24 DAgger training.

Like [src/oracle_24.py](src/oracle_24.py) but accepts arbitrary integer
targets. Given a `remaining` multiset and a `target`, returns the set of
winning next-ops whose resulting state can still reach the target.

Reuses OPS, apply_step, validate_step from oracle_24 where safe, but
reachability is now target-parameterized so we need our own cache.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Iterable

from src.oracle_24 import OPS, _apply, _canon, apply_step, validate_step  # noqa: F401


@lru_cache(maxsize=None)
def _can_reach(remaining: tuple, target: Fraction) -> bool:
    if len(remaining) == 1:
        return remaining[0] == target
    seen: set = set()
    for i in range(len(remaining)):
        for j in range(len(remaining)):
            if i == j:
                continue
            a, b = remaining[i], remaining[j]
            rest = tuple(remaining[k] for k in range(len(remaining))
                         if k != i and k != j)
            for sym, fn, commutative in OPS:
                if commutative and a > b:
                    continue
                r = fn(a, b)
                if r is None:
                    continue
                key = (sym, a, b, _canon(rest + (r,)))
                if key in seen:
                    continue
                seen.add(key)
                if _can_reach(_canon(rest + (r,)), target):
                    return True
    return False


def can_reach(remaining: Iterable[Fraction], target: int) -> bool:
    return _can_reach(_canon(remaining), Fraction(int(target)))


def winning_ops(remaining: Iterable[Fraction], target: int
                ) -> list[tuple[str, Fraction, Fraction, Fraction]]:
    """Return winning (op_sym, a, b, r) applicable to `remaining` toward
    `target`. Mirrors oracle_24.winning_ops but target-parameterized."""
    rem = _canon(remaining)
    tgt = Fraction(int(target))
    if len(rem) == 1:
        return []  # terminal
    if len(rem) < 2:
        return []
    winners: list[tuple[str, Fraction, Fraction, Fraction]] = []
    seen_keys: set = set()
    n = len(rem)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = rem[i], rem[j]
            rest = tuple(rem[k] for k in range(n) if k != i and k != j)
            for sym, fn, commutative in OPS:
                r = fn(a, b)
                if r is None:
                    continue
                new_state = _canon(rest + (r,))
                if not _can_reach(new_state, tgt):
                    continue
                key = (sym, a, b, r)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                winners.append((sym, a, b, r))
    return winners
