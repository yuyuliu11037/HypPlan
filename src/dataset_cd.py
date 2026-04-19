"""Dataset for Countdown CoT-SFT: prompt+completion with loss masking."""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset


INSTRUCTION = (
    "Use the six given numbers and integer arithmetic (+, -, *, /) to reach "
    "the target. Each number must be used exactly once. Subtraction must be "
    "non-negative. Division must be exact (no remainder)."
)


def make_prompt(pool: list[int], target: int) -> str:
    pool_str = " ".join(str(n) for n in sorted(pool))
    return f"{INSTRUCTION}\n\nProblem: {pool_str} | Target: {target}\nStep 1:"


def make_completion(text: str) -> str:
    """Extract completion (everything after the first 'Step 1:')."""
    marker = "Step 1:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx + len(marker):]


class CountdownSFTDataset(Dataset):
    """SFT dataset for Countdown.

    Each sample has input_ids, attention_mask, and labels. Labels are -100
    on prompt tokens and token ids on completion tokens.
    """

    def __init__(self, tokenizer, data_path: str, max_seq_len: int = 384):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        prompt = make_prompt(item["pool"], item["target"])
        completion = make_completion(item["text"])

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        if self.tokenizer.eos_token_id is not None:
            completion_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + completion_ids
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        labels = [-100] * len(prompt_ids) + completion_ids
        labels = labels[:len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad = max_len - b["input_ids"].size(0)
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }
