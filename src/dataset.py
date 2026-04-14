"""Dataset: loads EleutherAI/hendrycks_math, tokenizes with step boundary tracking."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

from src.utils import split_steps


class MathDataset(Dataset):
    """Dataset for Stage 1 training on MATH.

    Each sample is one MATH problem with its ground-truth solution, split into
    reasoning steps by sentence-ending periods. Tracks step boundary positions
    for planning vector insertion.
    """

    def __init__(self, tokenizer, split: str = "train", max_seq_len: int = 2048,
                 configs: list[str] | None = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer.
            split: "train" or "test".
            max_seq_len: Maximum sequence length.
            configs: List of hendrycks_math subject configs to load.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if configs is None:
            configs = ["algebra", "counting_and_probability", "geometry",
                       "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

        all_ds = []
        for cfg in configs:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
            all_ds.append(ds)
        self.data = concatenate_datasets(all_ds)

        # Skip examples with <= 1 step (no boundaries to train on)
        valid_indices = []
        for i in range(len(self.data)):
            steps = split_steps(self.data[i]["solution"])
            if len(steps) > 1:
                valid_indices.append(i)
        self.data = self.data.select(valid_indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item["problem"]
        solution = item["solution"]
        steps = split_steps(solution)
        return self._tokenize(problem, steps)

    def _tokenize(self, problem: str, steps: list[str]):
        """Tokenize problem + steps with step boundary tracking.

        Format: <problem>\n\n<step1>. <step2>. ... <stepN><eos>
        """
        problem_ids = self.tokenizer.encode(problem, add_special_tokens=False)
        sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)

        all_ids = list(problem_ids) + list(sep_ids)
        boundary_positions = []

        for i, step in enumerate(steps):
            if i > 0:
                dot_ids = self.tokenizer.encode(". ", add_special_tokens=False)
                all_ids.extend(dot_ids)

            boundary_positions.append(len(all_ids) - 1)

            step_ids = self.tokenizer.encode(step, add_special_tokens=False)
            all_ids.extend(step_ids)

        # Add EOS
        if self.tokenizer.eos_token_id is not None:
            all_ids.append(self.tokenizer.eos_token_id)

        # Truncate
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[:self.max_seq_len]

        # Filter positions within truncated sequence
        boundary_positions = [p for p in boundary_positions if p < len(all_ids)]

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(all_ids), dtype=torch.long),
            "boundary_positions": torch.tensor(boundary_positions, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate with padding."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    max_steps = max(b["boundary_positions"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    boundary_positions = []

    for b in batch:
        L = b["input_ids"].size(0)
        pad_len = max_len - L
        S = b["boundary_positions"].size(0)
        step_pad = max_steps - S

        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        boundary_positions.append(torch.cat([
            b["boundary_positions"],
            torch.full((step_pad,), -1, dtype=torch.long),
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "boundary_positions": torch.stack(boundary_positions),
    }
