from __future__ import annotations

import re

import sympy as sp


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize(expr: str) -> str:
    expr = expr.strip()
    if expr.endswith("."):
        expr = expr[:-1]
    return expr


def grade_answer(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    if pred_n == gold_n:
        return True
    try:
        p = sp.sympify(pred_n)
        g = sp.sympify(gold_n)
        return bool(sp.simplify(p - g) == 0)
    except Exception:  # noqa: BLE001
        return False

