from fractions import Fraction
from itertools import combinations

OPS = [
    ('+', lambda a, b: a + b, True),   # commutative
    ('-', lambda a, b: a - b, False),
    ('*', lambda a, b: a * b, True),
    ('/', lambda a, b: a / b if b != 0 else None, False),
]

def solve(nums):
    """Return list of trajectories. Each trajectory is a list of (a, op, b, result) steps."""
    nums = tuple(sorted(nums))  # canonical form for dedup
    if len(nums) == 1:
        return [[]] if nums[0] == Fraction(24) else []
    
    trajectories = []
    seen_pairs = set()
    for i, j in combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        remaining = tuple(nums[k] for k in range(len(nums)) if k != i and k != j)
        
        for op_sym, op_fn, commutative in OPS:
            # Try both orders for non-commutative ops
            orderings = [(a, b)] if commutative else [(a, b), (b, a)]
            for x, y in orderings:
                result = op_fn(x, y)
                if result is None:
                    continue
                new_nums = remaining + (result,)
                # Dedup: skip if we've tried this exact (op, operands, remaining) before
                key = (op_sym, x, y, tuple(sorted(remaining)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                
                for sub_traj in solve(new_nums):
                    trajectories.append([(x, op_sym, y, result)] + sub_traj)
    
    return trajectories

from itertools import combinations_with_replacement

all_problems = list(combinations_with_replacement(range(1, 14), 4))
# len = 1820

solved = {}
for prob in all_problems:
    nums = tuple(Fraction(n) for n in prob)
    trajs = solve(nums)
    if trajs:
        solved[prob] = trajs

import json
import re


def fraction_to_str(f):
    return str(int(f)) if f.denominator == 1 else str(f)


def trajectory_to_string(prob, traj):
    """Convert a problem key and trajectory into formatted text.

    Args:
        prob: string like "4,7,8,8"
        traj: list of step strings like ["8 / 8 = 1", "7 - 1 = 6", "4 * 6 = 24"]

    Returns:
        Formatted string with step-by-step solution.
    """
    pool = [Fraction(n) for n in prob.split(",")]
    lines = ["Problem: " + " ".join(str(int(n)) for n in pool)]

    for i, step in enumerate(traj):
        lhs, rhs = step.split(" = ")
        parts = re.split(r' ([+\-*/]) ', lhs)
        a_str, op, b_str = parts[0], parts[1], parts[2]
        a = Fraction(a_str)
        b = Fraction(b_str)
        result = Fraction(rhs)

        pool.remove(a)
        pool.remove(b)
        pool.append(result)

        if i < len(traj) - 1:
            remaining = " ".join(str(int(x)) if x.denominator == 1 else str(x)
                                 for x in sorted(pool))
            lines.append(f"Step {i+1}: {step}. Remaining: {remaining}")
        else:
            lines.append(f"Step {i+1}: {step}. Answer: 24")

    return "\n".join(lines)


def split_into_steps(text):
    """Return character offsets where each 'Step N:' begins.

    These are the positions where planning vectors t_i should be injected
    during training and inference.

    Args:
        text: formatted trajectory string from trajectory_to_string()

    Returns:
        list[int]: character offsets of each step boundary.
    """
    return [m.start() for m in re.finditer(r'^Step \d+:', text, re.MULTILINE)]

import os
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

output = {}
for prob, trajs in solved.items():
    key = ",".join(str(n) for n in prob)
    output[key] = [
        [f"{fraction_to_str(a)} {op} {fraction_to_str(b)} = {fraction_to_str(r)}" for a, op, b, r in traj]
        for traj in trajs
    ]

with open(os.path.join(DATA_DIR, "24_trajectories.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {sum(len(t) for t in output.values())} trajectories for {len(output)} solvable problems.")

import random
random.seed(42)

sampled = {}
for key, trajs in output.items():
    if len(trajs) <= 10:
        sampled[key] = trajs
    else:
        sampled[key] = random.sample(trajs, 10)

with open(os.path.join(DATA_DIR, "24_trajectories_sampled.json"), "w") as f:
    json.dump(sampled, f, indent=2)

print(f"Saved {sum(len(t) for t in sampled.values())} sampled trajectories for {len(sampled)} solvable problems.")

# Convert sampled trajectories to text format
text_output = {}
for key, trajs in sampled.items():
    text_output[key] = [trajectory_to_string(key, traj) for traj in trajs]

with open(os.path.join(DATA_DIR, "24_trajectories_text.json"), "w") as f:
    json.dump(text_output, f, indent=2)

print(f"Saved {sum(len(t) for t in text_output.values())} text trajectories to 24_trajectories_text.json")

# Split by problem into train/val/test (80/10/10)
problems = list(text_output.keys())
random.shuffle(problems)

n = len(problems)
n_val = n // 10
n_test = n // 10
n_train = n - n_val - n_test

train_probs = problems[:n_train]
val_probs = problems[n_train:n_train + n_val]
test_probs = problems[n_train + n_val:]

# Write JSONL splits — one line per trajectory, grouped by problem
for split_name, split_probs in [("train", train_probs), ("val", val_probs), ("test", test_probs)]:
    count = 0
    with open(os.path.join(DATA_DIR, f"24_{split_name}.jsonl"), "w") as f:
        for prob in split_probs:
            for text in text_output[prob]:
                offsets = split_into_steps(text)
                line = json.dumps({"problem": prob, "text": text, "step_offsets": offsets})
                f.write(line + "\n")
                count += 1
    print(f"{split_name}: {len(split_probs)} problems, {count} trajectories")