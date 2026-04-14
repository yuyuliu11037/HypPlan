"""Dataset for planning vector training on Game of 24.

Tokenizes prompt+trajectory and computes token-level boundary positions
for planning vector insertion.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from src.dataset_24 import make_prompt


class Game24PlanDataset(Dataset):
    """Dataset for planning vector training.

    Returns input_ids, attention_mask, and boundary_positions (token-level).
    """

    def __init__(self, tokenizer, data_path: str, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build full text: prompt + completion (same as SFT but as one string)
        prompt = make_prompt(item["problem"])
        # The text field starts with "Problem: ..." — extract completion after "Step 1:"
        full_text = item["text"]
        marker = "Step 1:"
        marker_idx = full_text.find(marker)
        completion = full_text[marker_idx + len(marker):] if marker_idx >= 0 else full_text

        # Full sequence: prompt + completion
        full_str = prompt + completion

        # Tokenize the full string
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            completion_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + completion_ids

        # Compute boundary positions: token index of the last token BEFORE each step starts
        # The prompt ends with "Step 1:", so the first boundary is at len(prompt_ids) - 1
        # For subsequent steps, we find "Step N:" in the token stream
        #
        # Strategy: tokenize prompt+"<step1_content>\nStep 2:" prefix to find where
        # each step boundary falls in token space.
        #
        # Simpler approach: use the character-level step_offsets from the data,
        # map each to the corresponding token position.
        boundary_positions = []

        # The first boundary is right before the completion starts (end of prompt)
        boundary_positions.append(len(prompt_ids) - 1)

        # For steps 2, 3: find their character offset in completion, tokenize prefix up to there
        step_offsets = item["step_offsets"]
        for offset in step_offsets[1:]:  # skip step 1 (already handled)
            # offset is character position in full_text where "Step N:" starts
            # Map to position in completion string
            comp_offset = offset - (marker_idx + len(marker))
            if comp_offset <= 0:
                continue
            # Tokenize completion up to this offset to find token position
            prefix_comp = completion[:comp_offset]
            prefix_comp_ids = self.tokenizer.encode(prefix_comp, add_special_tokens=False)
            # Boundary is at prompt_len + len(prefix_comp_ids) - 1
            bp = len(prompt_ids) + len(prefix_comp_ids) - 1
            if bp < len(input_ids):
                boundary_positions.append(bp)

        # Truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        boundary_positions = [p for p in boundary_positions if p < len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "boundary_positions": torch.tensor(boundary_positions, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad batch. boundary_positions padded with -1."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    max_bp = max(b["boundary_positions"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    boundary_positions = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))

        bp_pad = max_bp - b["boundary_positions"].size(0)
        boundary_positions.append(
            torch.cat([b["boundary_positions"], torch.full((bp_pad,), -1, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "boundary_positions": torch.stack(boundary_positions),
    }
