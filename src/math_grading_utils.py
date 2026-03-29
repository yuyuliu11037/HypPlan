"""Utilities for extracting and comparing math answers in \\boxed{} format."""

import re
import sympy
from sympy.parsing.latex import parse_latex


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    # Find all \boxed{ occurrences and take the last one
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    # Walk forward from the opening brace to find the matching close
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


def _strip_text_wrappers(s: str) -> str:
    """Remove common LaTeX text wrappers."""
    for cmd in ["\\text", "\\mathrm", "\\textbf", "\\mathbf", "\\mbox"]:
        # e.g. \text{hello} -> hello
        while cmd + "{" in s:
            start = s.index(cmd + "{")
            # find matching brace
            depth = 0
            j = start + len(cmd)
            for k in range(j, len(s)):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        inner = s[j + 1 : k]
                        s = s[:start] + inner + s[k + 1 :]
                        break
    return s


def normalize_answer(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    if answer is None:
        return ""

    s = answer.strip()

    # Remove dollar signs
    s = s.replace("$", "")

    # Remove \left and \right
    s = s.replace("\\left", "").replace("\\right", "")

    # Remove text wrappers
    s = _strip_text_wrappers(s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Remove trailing period
    s = s.rstrip(".")

    # Normalize common LaTeX
    s = s.replace("\\infty", "oo")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\%", "")
    s = s.replace("%", "")
    s = s.replace("\\$", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\quad", " ")

    # dfrac -> frac
    s = s.replace("\\dfrac", "\\frac")
    # tfrac -> frac
    s = s.replace("\\tfrac", "\\frac")

    return s.strip()


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number."""
    s = s.replace(",", "")  # remove thousands separators
    try:
        return float(s)
    except ValueError:
        return None


def _try_sympy_equal(a: str, b: str) -> bool | None:
    """Try to check equality using sympy. Returns None if parsing fails."""
    try:
        expr_a = parse_latex(a)
        expr_b = parse_latex(b)
        diff = sympy.simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        return None


def is_equiv(pred: str, target: str) -> bool:
    """Check if two math answers are equivalent."""
    if pred is None or target is None:
        return False

    pred_n = normalize_answer(pred)
    target_n = normalize_answer(target)

    if not pred_n or not target_n:
        return False

    # Direct string match after normalization
    if pred_n == target_n:
        return True

    # Try numeric comparison
    pred_num = _try_parse_number(pred_n)
    target_num = _try_parse_number(target_n)
    if pred_num is not None and target_num is not None:
        return abs(pred_num - target_num) < 1e-6

    # Try sympy symbolic comparison
    result = _try_sympy_equal(pred_n, target_n)
    if result is not None:
        return result

    return False
