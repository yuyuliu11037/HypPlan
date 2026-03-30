"""Grade sampled solutions and filter to problems with pass rate in [0.2, 0.8]."""

import json
from collections import Counter

from math_grading_utils import extract_boxed_answer, is_equiv

INPUT_PATH = "results/math_samples.jsonl"
GRADED_PATH = "results/math_graded.jsonl"
FILTERED_PATH = "results/math_filtered.jsonl"

PASS_RATE_LOW = 0.1
PASS_RATE_HIGH = 0.9


def grade_problem(record: dict) -> dict:
    """Grade all generations for a single problem."""
    target_answer = extract_boxed_answer(record["solution"])
    correct = []
    for gen in record["generations"]:
        pred_answer = extract_boxed_answer(gen)
        correct.append(is_equiv(pred_answer, target_answer))

    pass_rate = sum(correct) / len(correct)
    return {**record, "correct": correct, "pass_rate": pass_rate}


def print_statistics(graded: list[dict], filtered: list[dict]):
    """Print summary statistics."""
    total = len(graded)
    pass_rates = [r["pass_rate"] for r in graded]

    print(f"\n{'='*60}")
    print(f"Total problems: {total}")
    print(f"Average pass rate: {sum(pass_rates)/len(pass_rates):.3f}")
    print()

    # Pass rate distribution (histogram in 0.1 buckets)
    buckets = Counter()
    for pr in pass_rates:
        bucket = min(int(pr * 10), 9) / 10  # [0.0, 0.1, ..., 0.9]
        buckets[bucket] += 1

    print("Pass rate distribution:")
    for bucket in sorted(buckets.keys()):
        count = buckets[bucket]
        bar = "#" * (count // 10)
        print(f"  [{bucket:.1f}, {bucket+0.1:.1f}): {count:5d}  {bar}")

    print()

    # Filtered stats
    print(f"Filtered to pass rate in [{PASS_RATE_LOW}, {PASS_RATE_HIGH}]: {len(filtered)} problems")

    # Breakdown by subject
    type_counts = Counter(r["type"] for r in filtered)
    print("\nFiltered problems by subject:")
    for subject, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {subject}: {count}")

    # Breakdown by level
    level_counts = Counter(r["level"] for r in filtered)
    print("\nFiltered problems by level:")
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")

    print(f"{'='*60}\n")


def main():
    # Load sampled generations
    with open(INPUT_PATH) as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} problems from {INPUT_PATH}")

    # Grade all problems
    graded = []
    for i, record in enumerate(records):
        graded.append(grade_problem(record))
        if (i + 1) % 500 == 0:
            print(f"  Graded {i+1}/{len(records)}")
    print(f"Graded all {len(graded)} problems")

    # Save graded results
    with open(GRADED_PATH, "w") as f:
        for record in graded:
            f.write(json.dumps(record) + "\n")
    print(f"Saved graded results to {GRADED_PATH}")

    # Filter by pass rate
    filtered = [r for r in graded if PASS_RATE_LOW <= r["pass_rate"] <= PASS_RATE_HIGH]

    # Save filtered results
    with open(FILTERED_PATH, "w") as f:
        for record in filtered:
            f.write(json.dumps(record) + "\n")
    print(f"Saved filtered results to {FILTERED_PATH}")

    print_statistics(graded, filtered)


if __name__ == "__main__":
    main()
