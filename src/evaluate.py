"""Evaluation: extract answers and compute accuracy."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from src.utils import extract_boxed_answer, normalize_answer


def is_correct(predicted: str | None, target: str | None) -> bool:
    """Check if predicted answer matches target (exact match after normalization)."""
    if predicted is None or target is None:
        return False
    return normalize_answer(predicted) == normalize_answer(target)


def evaluate(generations_path: str) -> dict:
    """Evaluate generated solutions."""
    with open(generations_path) as f:
        records = [json.loads(line) for line in f]

    total = 0
    correct = 0
    by_level = defaultdict(lambda: {"total": 0, "correct": 0})
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})

    for record in records:
        pred_answer = extract_boxed_answer(record["generation"])
        target_answer = extract_boxed_answer(record["solution"])

        hit = is_correct(pred_answer, target_answer)
        total += 1
        correct += int(hit)

        level = record.get("level", "Unknown")
        by_level[level]["total"] += 1
        by_level[level]["correct"] += int(hit)

        problem_type = record.get("type", "Unknown")
        by_type[problem_type]["total"] += 1
        by_type[problem_type]["correct"] += int(hit)

    accuracy = correct / total if total > 0 else 0.0

    results = {
        "overall": {"accuracy": accuracy, "correct": correct, "total": total},
        "by_level": {},
        "by_type": {},
    }

    for level in sorted(by_level.keys(), key=str):
        stats = by_level[level]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        results["by_level"][str(level)] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    for ptype in sorted(by_type.keys()):
        stats = by_type[ptype]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        results["by_type"][ptype] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    overall = results["overall"]
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {overall['accuracy']:.4f} ({overall['correct']}/{overall['total']})")

    print(f"\nBy Level:")
    for level, stats in results["by_level"].items():
        print(f"  Level {level}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    print(f"\nBy Type:")
    for ptype, stats in results["by_type"].items():
        print(f"  {ptype}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    print(f"{'='*60}\n")


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
