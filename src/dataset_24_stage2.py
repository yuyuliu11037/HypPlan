"""Dataset for stage-2 training.

Extends Game24PlanDataset with per-boundary canonical state text. At each
step boundary, we need the state-before-next-step rendered in the same format
the stage-1 head was trained on — so forwarding it through the frozen SFT
base gives the head an in-distribution input.
"""
from __future__ import annotations

import json
import re
from fractions import Fraction

import torch
from torch.utils.data import Dataset

from src.dataset_24 import make_prompt
from src.tree_data import render_state_from_history


STEP_RE = re.compile(r"Step\s+\d+:\s+(\S+)\s+([+\-*/])\s+(\S+)\s+=\s+(\S+)")
MAX_STATE_TOKENS = 128


def parse_trajectory(text: str) -> list:
    """Extract list of (a, op, b, r) tuples from a formatted trajectory."""
    steps = []
    for m in STEP_RE.finditer(text):
        a = Fraction(m.group(1))
        op = m.group(2)
        b = Fraction(m.group(3))
        r_str = m.group(4).rstrip(".")
        r = Fraction(r_str)
        steps.append((a, op, b, r))
    return steps


class Game24Stage2Dataset(Dataset):
    """Yields the main prompt+completion tokenization plus one pre-tokenized
    canonical state per step boundary.

    Fields returned per sample:
        input_ids, attention_mask, boundary_positions  -- as Game24PlanDataset
        state_input_ids: list[tensor], len == num_boundaries. Each tensor is
            the tokenized canonical state text BEFORE the corresponding step.
        state_attention_mask: list[tensor], same shape.
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
        problem = item["problem"]
        full_text = item["text"]

        # --- Main sequence tokenization (mirrors Game24PlanDataset) ---
        prompt = make_prompt(problem)
        marker = "Step 1:"
        marker_idx = full_text.find(marker)
        completion = full_text[marker_idx + len(marker):] if marker_idx >= 0 else full_text

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            completion_ids.append(self.tokenizer.eos_token_id)
        input_ids = prompt_ids + completion_ids

        # --- Boundary positions (same logic as Game24PlanDataset) ---
        boundary_positions = [len(prompt_ids) - 1]
        step_offsets = item["step_offsets"]
        for offset in step_offsets[1:]:
            comp_offset = offset - (marker_idx + len(marker))
            if comp_offset <= 0:
                continue
            prefix_comp_ids = self.tokenizer.encode(
                completion[:comp_offset], add_special_tokens=False,
            )
            bp = len(prompt_ids) + len(prefix_comp_ids) - 1
            if bp < len(input_ids):
                boundary_positions.append(bp)

        # --- Truncate main sequence, drop late boundaries if needed ---
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        boundary_positions = [p for p in boundary_positions if p < len(input_ids)]

        # --- Canonical state tokenization per boundary ---
        steps = parse_trajectory(full_text)
        state_input_ids: list = []
        for i in range(len(boundary_positions)):
            history = tuple(steps[:i])
            text_i = render_state_from_history(problem, history)
            ids = self.tokenizer.encode(text_i, add_special_tokens=True)
            if len(ids) > MAX_STATE_TOKENS:
                ids = ids[:MAX_STATE_TOKENS]
            state_input_ids.append(torch.tensor(ids, dtype=torch.long))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "boundary_positions": torch.tensor(boundary_positions, dtype=torch.long),
            "state_input_ids": state_input_ids,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad main sequences + stack states per-boundary position.

    Returns:
        input_ids: (B, L_max)
        attention_mask: (B, L_max)
        boundary_positions: (B, K_max) padded with -1
        state_input_ids: list of length K_max, each (B, S_k) padded with 0
        state_attention_mask: list matching shape of state_input_ids
        state_valid: (B, K_max) bool — True where the sample has that boundary
    """
    max_len = max(b["input_ids"].size(0) for b in batch)
    max_bp = max(b["boundary_positions"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    boundary_positions = []
    state_valid = torch.zeros(len(batch), max_bp, dtype=torch.bool)

    # Per boundary index k, find the max state length across samples that have it
    max_state_lens = [0] * max_bp

    for bi, b in enumerate(batch):
        pad = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"],
                                          torch.zeros(pad, dtype=torch.long)]))
        bp = b["boundary_positions"]
        bp_pad = max_bp - bp.size(0)
        boundary_positions.append(
            torch.cat([bp, torch.full((bp_pad,), -1, dtype=torch.long)])
        )
        for k in range(bp.size(0)):
            state_valid[bi, k] = True
            s_len = b["state_input_ids"][k].size(0)
            if s_len > max_state_lens[k]:
                max_state_lens[k] = s_len

    # Build per-boundary batched state tensors (B, S_k)
    state_ids_per_k: list = []
    state_mask_per_k: list = []
    for k in range(max_bp):
        Sk = max_state_lens[k]
        ids = torch.zeros(len(batch), max(Sk, 1), dtype=torch.long)
        mask = torch.zeros(len(batch), max(Sk, 1), dtype=torch.long)
        for bi, b in enumerate(batch):
            if k < len(b["state_input_ids"]):
                s = b["state_input_ids"][k]
                ids[bi, :s.size(0)] = s
                mask[bi, :s.size(0)] = 1
        state_ids_per_k.append(ids)
        state_mask_per_k.append(mask)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "boundary_positions": torch.stack(boundary_positions),
        "state_input_ids": state_ids_per_k,     # list length K_max, each (B, S_k)
        "state_attention_mask": state_mask_per_k,
        "state_valid": state_valid,
    }
