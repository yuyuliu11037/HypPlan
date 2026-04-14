"""Evaluation for Game of 24: validate generated solutions."""
from __future__ import annotations

import argparse
import json
import os
import re
from fractions import Fraction


def parse_and_validate(problem: str, generation: str) -> bool:
    """Check if generation contains a valid 3-step solution reaching 24.

    Validates:
    - Each arithmetic step is correct
    - Only the given numbers are used, each exactly once
    - Final result is 24
    """
    nums = [Fraction(n) for n in problem.split(",")]
    pool = list(nums)

    # Extract steps: "Step N: a op b = result"
    step_pattern = re.compile(
        r'Step\s+\d+:\s*(-?[\d./]+)\s*([+\-*/])\s*(-?[\d./]+)\s*=\s*(-?[\d./]+)'
    )
    steps = step_pattern.findall(generation)

    if len(steps) != 3:
        return False

    for a_str, op, b_str, r_str in steps:
        try:
            a = Fraction(a_str)
            b = Fraction(b_str)
            r = Fraction(r_str)
        except (ValueError, ZeroDivisionError):
            return False

        # Check operands are in the pool
        try:
            pool.remove(a)
            pool.remove(b)
        except ValueError:
            return False

        # Verify arithmetic
        if op == '+':
            expected = a + b
        elif op == '-':
            expected = a - b
        elif op == '*':
            expected = a * b
        elif op == '/':
            if b == 0:
                return False
            expected = a / b
        else:
            return False

        if r != expected:
            return False

        pool.append(r)

    # Final pool should be exactly [24]
    return len(pool) == 1 and pool[0] == Fraction(24)


def evaluate(generations_path: str) -> dict:
    """Evaluate Game of 24 generations."""
    with open(generations_path) as f:
        records = [json.loads(line) for line in f]

    total = 0
    correct = 0
    details = []

    for record in records:
        valid = parse_and_validate(record["problem"], record["generation"])
        total += 1
        correct += int(valid)
        details.append({
            "problem": record["problem"],
            "valid": valid,
            "generation": record["generation"][:200],
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "overall": {"accuracy": accuracy, "correct": correct, "total": total},
        "details": details,
    }


def print_results(results: dict):
    overall = results["overall"]
    print(f"\n{'='*60}")
    print(f"Game of 24 — Zero-shot Accuracy: {overall['accuracy']:.4f} "
          f"({overall['correct']}/{overall['total']})")
    print(f"{'='*60}")

    # Show a few examples
    correct_examples = [d for d in results["details"] if d["valid"]][:3]
    wrong_examples = [d for d in results["details"] if not d["valid"]][:3]

    if correct_examples:
        print("\nCorrect examples:")
        for d in correct_examples:
            print(f"  [{d['problem']}] {d['generation'][:100]}")

    if wrong_examples:
        print("\nIncorrect examples:")
        for d in wrong_examples:
            print(f"  [{d['problem']}] {d['generation'][:100]}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with generations")
    parser.add_argument("--output", default=None, help="Output JSON for metrics")
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
