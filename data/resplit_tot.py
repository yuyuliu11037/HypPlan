"""Re-split Game of 24 data to match Tree-of-Thoughts test set.

Test: 100 puzzles from 4nums.com rank 901-1000 (same as ToT paper).
Val: ~50 problems (small, for efficiency).
SFT train: ~400 problems.
Plan train: remaining (~812 problems), zero overlap with SFT.

All splits are at the problem level (no problem appears in multiple splits).
SFT and plan pools have zero problem overlap.
"""
import csv
import json
import os
import random
import re
from fractions import Fraction

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Step 1: Load ToT test set (rank 901-1000 from 4nums.com) ─────────────────
tot_test_problems = set()
with open(os.path.join(DATA_DIR, "4nums_ranked.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rank = int(row["Rank"])
        if 901 <= rank <= 1000:
            nums = tuple(sorted(int(x) for x in row["Puzzles"].split()))
            tot_test_problems.add(nums)

print(f"ToT test problems (rank 901-1000): {len(tot_test_problems)}")

# ── Step 2: Load all trajectories (text format) ──────────────────────────────
with open(os.path.join(DATA_DIR, "24_trajectories_text.json")) as f:
    text_output = json.load(f)

print(f"Total solvable problems with trajectories: {len(text_output)}")


def split_into_steps(text):
    """Return character offsets where each 'Step N:' begins."""
    return [m.start() for m in re.finditer(r'^Step \d+:', text, re.MULTILINE)]


# ── Step 3: Separate test from rest ──────────────────────────────────────────
def prob_str_to_tuple(s):
    return tuple(sorted(int(x) for x in s.split(",")))


test_probs = []
rest_probs = []
for prob_str in text_output:
    tup = prob_str_to_tuple(prob_str)
    if tup in tot_test_problems:
        test_probs.append(prob_str)
    else:
        rest_probs.append(prob_str)

print(f"Test: {len(test_probs)}, Rest: {len(rest_probs)}")
assert len(test_probs) == 100, f"Expected 100 test problems, got {len(test_probs)}"

# ── Step 4: Split rest into val / SFT / plan ─────────────────────────────────
random.shuffle(rest_probs)

n_val = 50
n_sft = 400
n_plan = len(rest_probs) - n_val - n_sft

val_probs = rest_probs[:n_val]
sft_probs = rest_probs[n_val:n_val + n_sft]
plan_probs = rest_probs[n_val + n_sft:]

print(f"Val: {len(val_probs)}, SFT: {len(sft_probs)}, Plan: {len(plan_probs)}")

# Verify zero overlap
sets = {
    "test": set(test_probs),
    "val": set(val_probs),
    "sft": set(sft_probs),
    "plan": set(plan_probs),
}
for a_name, a_set in sets.items():
    for b_name, b_set in sets.items():
        if a_name < b_name:
            overlap = a_set & b_set
            assert len(overlap) == 0, f"Overlap between {a_name} and {b_name}: {overlap}"
print("✓ Zero overlap verified between all splits")

# ── Step 5: Write JSONL files ─────────────────────────────────────────────────

def write_jsonl(probs, filename, max_trajs=None):
    """Write JSONL with optional trajectory sampling."""
    all_trajs = []
    for prob in probs:
        for text in text_output[prob]:
            offsets = split_into_steps(text)
            all_trajs.append({"problem": prob, "text": text, "step_offsets": offsets})

    if max_trajs and len(all_trajs) > max_trajs:
        all_trajs = random.sample(all_trajs, max_trajs)

    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        for item in all_trajs:
            f.write(json.dumps(item) + "\n")

    # Count unique problems in output
    unique_probs = set(item["problem"] for item in all_trajs)
    print(f"  {filename}: {len(all_trajs)} trajectories, {len(unique_probs)} unique problems")
    return all_trajs


# Test: all trajectories (for reference/evaluation)
write_jsonl(test_probs, "24_test_tot.jsonl")

# Val: all trajectories
write_jsonl(val_probs, "24_val_tot.jsonl")

# SFT: sample 3000 trajectories
write_jsonl(sft_probs, "24_train_sft3k_tot.jsonl", max_trajs=3000)

# Plan: sample 5000 trajectories
write_jsonl(plan_probs, "24_train_plan5k_tot.jsonl", max_trajs=5000)

print("\nDone! New data files written.")
