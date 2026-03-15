from __future__ import annotations

import re
from typing import Optional

from sympy import simplify, sympify


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text: str) -> Optional[str]:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    return text


def grade_answer(predicted: str | None, ground_truth: str | None) -> bool:
    if predicted is None or ground_truth is None:
        return False

    p = _normalize_answer(predicted)
    g = _normalize_answer(ground_truth)
    if p == g:
        return True

    try:
        return simplify(sympify(p) - sympify(g)) == 0
    except Exception:
        return False
