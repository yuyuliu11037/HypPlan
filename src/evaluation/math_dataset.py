"""Load the MATH dataset for evaluation.

Uses qwedsacf/competition_math on the Hugging Face Hub (same MATH content
and schema as the original hendrycks/competition_math: problem, solution, level, type).
"""

from __future__ import annotations

from datasets import load_dataset


def load_math_test():
    """Load the MATH dataset from qwedsacf/competition_math (single split, 12.5k rows)."""
    return load_dataset("qwedsacf/competition_math", split="train")
