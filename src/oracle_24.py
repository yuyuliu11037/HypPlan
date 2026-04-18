"""Oracle for Game-of-24 DAgger training.

Given a `remaining` multiset of `Fraction`s, returns the set of "winning"
next-operations — operations whose resulting state can still reach 24.

Also provides a state validator for invalid-step detection during rollout.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Iterable

# (op_symbol, fn, commutative)
OPS = [
    ("+", lambda a, b: a + b, True),
    ("-", lambda a, b: a - b, False),
    ("*", lambda a, b: a * b, True),
    ("/", lambda a, b: None if b == 0 else a / b, False),
]

TARGET = Fraction(24)


def _apply(op_sym: str, a: Fraction, b: Fraction):
    for sym, fn, _ in OPS:
        if sym == op_sym:
            return fn(a, b)
    return None


def _canon(remaining: Iterable[Fraction]) -> tuple:
    """Canonicalize a remaining multiset: sorted tuple of Fractions."""
    return tuple(sorted(Fraction(r) for r in remaining))


@lru_cache(maxsize=None)
def can_reach_24(remaining: tuple) -> bool:
    """True iff there exists some sequence of ops on `remaining` ending in 24.

    `remaining` must be a canonicalized tuple of Fractions (sorted).
    """
    if len(remaining) == 1:
        return remaining[0] == TARGET
    seen = set()
    for i in range(len(remaining)):
        for j in range(len(remaining)):
            if i == j:
                continue
            a, b = remaining[i], remaining[j]
            rest = tuple(remaining[k] for k in range(len(remaining)) if k != i and k != j)
            for sym, fn, commutative in OPS:
                if commutative and a > b:
                    continue  # canonicalize to one ordering for commutative ops
                r = fn(a, b)
                if r is None:
                    continue
                key = (sym, a, b, _canon(rest + (r,)))
                if key in seen:
                    continue
                seen.add(key)
                if can_reach_24(_canon(rest + (r,))):
                    return True
    return False


def winning_ops(remaining: Iterable[Fraction]) -> list[tuple[str, Fraction, Fraction, Fraction]]:
    """Return list of winning (op_sym, a, b, r) applicable to `remaining`.

    For commutative ops we include BOTH orderings (a, b) and (b, a) when
    a ≠ b, so that either textual realization counts as a winner under the
    log-of-sum loss. For non-commutative ops (-, /), only the legal
    ordering is emitted.

    Each tuple is such that (a, op, b) = r, applied to `remaining` (removing
    one instance each of a and b, adding r) gives a state that can reach 24.
    """
    rem = _canon(remaining)
    if len(rem) < 2:
        return []
    # Find all distinct (a, op, b, result, new_state) that are "winning".
    winners: list[tuple[str, Fraction, Fraction, Fraction]] = []
    seen_keys = set()
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
                if not can_reach_24(new_state):
                    continue
                key = (sym, a, b, r)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                winners.append((sym, a, b, r))
    return winners


def validate_step(remaining: Iterable[Fraction], a: Fraction, op: str, b: Fraction,
                  r: Fraction) -> tuple[bool, str]:
    """Check whether "a op b = r" is a legal step from `remaining`.

    Returns (is_valid, reason). is_valid=True means:
      - a and b both appear in the remaining multiset (with multiplicity)
      - op is one of +,-,*,/
      - a op b actually equals r (using exact Fraction arithmetic)
    """
    rem = list(_canon(remaining))
    a = Fraction(a); b = Fraction(b); r = Fraction(r)

    # Check multiset membership
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
        return False, f"op {op} undefined (e.g. division by zero)"
    if computed != r:
        return False, f"arithmetic mismatch: {a} {op} {b} = {computed} != {r}"
    return True, ""


def apply_step(remaining: Iterable[Fraction], a: Fraction, b: Fraction,
               r: Fraction) -> tuple:
    """Return the new canonical remaining after one step. Assumes validated."""
    rem = list(_canon(remaining))
    rem.remove(Fraction(a))
    rem.remove(Fraction(b))
    rem.append(Fraction(r))
    return _canon(rem)
