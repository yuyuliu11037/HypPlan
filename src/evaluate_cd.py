"""Evaluate Countdown generations.

Validates that the generation is an N-1-step solution:
- operands at each step are drawn from the current pool (multiset)
- each step uses legal integer arithmetic (+, -, *, /) with:
    * subtraction non-negative
    * division exact (no remainder)
- final result == target
"""
from __future__ import annotations

import argparse
import json
import os
import re


_STEP_RE = re.compile(
    r'Step\s+\d+:\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)'
)


def _apply(op: str, a: int, b: int):
    if op == "+":
        return a + b
    if op == "-":
        return a - b if a >= b else None
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a // b
    return None


def parse_and_validate(pool: list[int], target: int, generation: str,
                       n_expected_steps: int) -> bool:
    """True iff generation is a valid N-1 step solution reaching target."""
    current = list(sorted(pool))
    steps = _STEP_RE.findall(generation)
    if len(steps) != n_expected_steps:
        return False

    for a_str, op, b_str, r_str in steps:
        try:
            a, b, r = int(a_str), int(b_str), int(r_str)
        except ValueError:
            return False
        try:
            current.remove(a)
        except ValueError:
            return False
        try:
            current.remove(b)
        except ValueError:
            return False
        expected = _apply(op, a, b)
        if expected is None or expected != r:
            return False
        current.append(r)

    return len(current) == 1 and current[0] == target


def evaluate(generations_path: str) -> dict:
    with open(generations_path) as f:
        records = [json.loads(line) for line in f]

    total = 0
    correct = 0
    details = []
    for record in records:
        pool = record["pool"]
        n_steps = len(pool) - 1
        valid = parse_and_validate(pool, record["target"],
                                   record["generation"], n_steps)
        total += 1
        correct += int(valid)
        details.append({
            "pool": pool, "target": record["target"],
            "valid": valid,
            "generation": record["generation"][:300],
        })
    accuracy = correct / total if total else 0.0
    return {
        "overall": {"accuracy": accuracy, "correct": correct, "total": total},
        "details": details,
    }


def print_results(results: dict) -> None:
    o = results["overall"]
    print(f"\n{'='*60}")
    print(f"Countdown Accuracy: {o['accuracy']:.4f} "
          f"({o['correct']}/{o['total']})")
    print(f"{'='*60}")
    correct = [d for d in results["details"] if d["valid"]][:3]
    wrong = [d for d in results["details"] if not d["valid"]][:3]
    if correct:
        print("\nCorrect examples:")
        for d in correct:
            print(f"  [{d['pool']} -> {d['target']}] {d['generation'][:120]}")
    if wrong:
        print("\nIncorrect examples:")
        for d in wrong:
            print(f"  [{d['pool']} -> {d['target']}] {d['generation'][:120]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = evaluate(args.input)
    print_results(results)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
