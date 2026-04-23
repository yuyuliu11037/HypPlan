"""Evaluation for varied-target Game-of-24 (generic integer arithmetic task).

Like [src/evaluate_24.py](src/evaluate_24.py) but:
  - number of steps is variable (can be 0 for trivial pool==[target]),
  - target is read from the problem record, not hardcoded to 24,
  - pool is a list of positive integers instead of a 4-string.

Generic parser tolerates an optional 'Step N:' prefix and accepts either:
  - Chain form: zero or more "Step N: a op b = r" lines, ending in "Answer: T".
  - Trivial form: pool == [target] and the generation says "Answer: T".
"""
from __future__ import annotations

import re
from fractions import Fraction


_STEP_RE = re.compile(
    r"Step\s+\d+:\s*(-?[\d./]+)\s*([+\-*/])\s*(-?[\d./]+)\s*=\s*(-?[\d./]+)"
)
_ANSWER_RE = re.compile(r"Answer:\s*(-?[\d./]+)")


def parse_and_validate_generic(pool: list[int], target: int,
                               generation: str) -> bool:
    """Check if `generation` is a valid chain reaching `target` from `pool`.

    Rules:
      - Each step's math must check out (a op b = r).
      - Operands must be present in the current pool; remove them, append r.
      - Final pool after all steps must be [target].
      - 'Answer: T' line must match target.
      - Number of steps must equal len(pool) - 1 (but we accept 0 steps only
        if len(pool) == 1 and pool[0] == target).
    """
    answer_match = _ANSWER_RE.search(generation)
    if not answer_match:
        return False
    try:
        if Fraction(answer_match.group(1)) != Fraction(target):
            return False
    except (ValueError, ZeroDivisionError):
        return False

    steps = _STEP_RE.findall(generation)
    expected_n_steps = len(pool) - 1
    if len(steps) != expected_n_steps:
        return False

    current: list[Fraction] = [Fraction(n) for n in pool]

    for a_str, op, b_str, r_str in steps:
        try:
            a = Fraction(a_str)
            b = Fraction(b_str)
            r = Fraction(r_str)
        except (ValueError, ZeroDivisionError):
            return False

        try:
            current.remove(a)
            current.remove(b)
        except ValueError:
            return False

        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op == "*":
            expected = a * b
        elif op == "/":
            if b == 0:
                return False
            expected = a / b
        else:
            return False

        if r != expected:
            return False

        current.append(r)

    return len(current) == 1 and current[0] == Fraction(target)
