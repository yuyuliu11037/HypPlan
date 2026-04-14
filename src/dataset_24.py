"""Dataset for Game of 24 CoT-SFT: prompt+completion with loss masking."""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset


INSTRUCTION = (
    "Use the four given numbers and basic arithmetic operations "
    "(+, -, *, /) to obtain 24. Each number must be used exactly once."
)


def make_prompt(problem: str) -> str:
    """Build the prompt prefix (no completion)."""
    nums = problem.replace(",", " ")
    return f"{INSTRUCTION}\n\nProblem: {nums}\nStep 1:"


def make_completion(text: str) -> str:
    """Extract completion from the full trajectory text.

    text starts with 'Problem: ...\nStep 1: ...'. We want everything
    after the first 'Step 1:' prefix.
    """
    marker = "Step 1:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx + len(marker):]


class Game24SFTDataset(Dataset):
    """SFT dataset for Game of 24.

    Each sample has input_ids, attention_mask, and labels.
    Labels are -100 on prompt tokens (no loss) and token ids on completion tokens.
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
        prompt = make_prompt(item["problem"])
        completion = make_completion(item["text"])

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        # Append EOS
        if self.tokenizer.eos_token_id is not None:
            completion_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + completion_ids

        # Truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        # Labels: -100 for prompt, token ids for completion
        labels = [-100] * len(prompt_ids) + completion_ids
        labels = labels[:len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad batch to max length with right-padding."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }
