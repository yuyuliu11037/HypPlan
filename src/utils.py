"""Data utilities: step splitting, answer extraction, normalization."""
from __future__ import annotations

import re


def split_steps(text: str) -> list[str]:
    """Split solution text into reasoning steps on sentence-ending periods.

    Splits on periods NOT preceded by a digit (to preserve decimals like 3.14),
    followed by whitespace.
    """
    parts = re.split(r'(?<!\d)\.(?=\s)', text)
    return [p.strip() for p in parts if p.strip()]


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract content from the last \boxed{...} in text.

    Handles nested braces correctly.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[start : i - 1].strip()


def extract_boxed_span(text: str) -> tuple[int, int] | None:
    r"""Return (start, end) character indices of content inside last \boxed{...}.

    Returns indices such that text[start:end] is the content (excluding markup).
    Returns None if no \boxed{} found or braces are unbalanced.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return (start, i - 1)


def normalize_answer(answer: str) -> str:
    """Basic normalization for answer comparison."""
    if answer is None:
        return ""
    answer = answer.strip()
    answer = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.rstrip(".")
    return answer.strip()
