from __future__ import annotations

import re

import sympy as sp


def _extract_braced(text: str, open_brace_idx: int) -> tuple[str | None, int | None]:
    if open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None, None
    depth = 0
    chars: list[str] = []
    for idx in range(open_brace_idx, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), idx
            chars.append(ch)
        else:
            chars.append(ch)
    return None, None


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed"
    last_match: str | None = None
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        brace_idx = idx + len(marker)
        while brace_idx < len(text) and text[brace_idx].isspace():
            brace_idx += 1
        content, end_idx = _extract_braced(text, brace_idx)
        if content is not None:
            last_match = content.strip()
            start = end_idx + 1
        else:
            start = brace_idx + 1
    return last_match


def has_complete_boxed(text: str) -> bool:
    return extract_boxed(text) is not None


def _cleanup_candidate(text: str) -> str | None:
    cleaned = text.replace("<|endoftext|>", "").strip().strip("*").strip()
    if not cleaned:
        return None
    if cleaned in {"Solution:", "Problem:"}:
        return None
    return cleaned.rstrip(".").strip()


def extract_final_answer(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    lower = text.lower()
    for marker in ["# answer", "final answer:", "answer:"]:
        idx = lower.rfind(marker)
        if idx == -1:
            continue
        tail = text[idx + len(marker) :]
        for line in tail.splitlines():
            candidate = _cleanup_candidate(line)
            if candidate:
                return candidate

    for line in reversed(text.splitlines()):
        candidate = _cleanup_candidate(line)
        if candidate:
            return candidate
    return None


def _replace_latex_frac(expr: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    out = expr
    while True:
        new = pattern.sub(r"(\1)/(\2)", out)
        if new == out:
            return new
        out = new


def _normalize(expr: str) -> str:
    expr = expr.replace("<|endoftext|>", "").strip().strip("$")
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("^\\circ", "").replace("\\circ", "")
    expr = re.sub(r"\bdegrees?\b", "", expr, flags=re.IGNORECASE)
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    expr = expr.replace("\\%", "%")
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)
    expr = _replace_latex_frac(expr)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = re.sub(r"\s+", "", expr)
    if expr.endswith("."):
        expr = expr[:-1]
    if expr.endswith("%"):
        number = expr[:-1]
        try:
            return str(float(number) / 100)
        except ValueError:
            return expr
    return expr


def _choice_label(expr: str) -> str:
    expr = _normalize(expr)
    expr = re.sub(r"[^A-Za-z]", "", expr)
    return expr.upper()


def _candidate_forms(expr: str) -> list[str]:
    normalized = _normalize(expr)
    candidates = [normalized]
    if "=" in normalized:
        candidates.extend(part for part in normalized.split("=") if part)
    return list(dict.fromkeys(candidates))


def _sympy_equal(pred: str, gold: str) -> bool:
    try:
        p = sp.sympify(pred)
        g = sp.sympify(gold)
        return bool(sp.simplify(p - g) == 0)
    except Exception:  # noqa: BLE001
        return False


def grade_answer(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False

    pred_candidates = _candidate_forms(pred)
    gold_candidates = _candidate_forms(gold)
    if any(p == g for p in pred_candidates for g in gold_candidates):
        return True

    pred_choice = _choice_label(pred)
    gold_choice = _choice_label(gold)
    if pred_choice and pred_choice == gold_choice:
        return True

    for p in pred_candidates:
        for g in gold_candidates:
            if _sympy_equal(p, g):
                return True
    return False

