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


def make_completion_plan(text: str) -> str:
    """Completion extractor for planning-tokens trajectories.

    text starts with 'Problem: ...\n<PLAN:OP> Step 1: ...'. We want
    everything after 'Problem: ...\\n' — i.e. starting with the first
    <PLAN:OP> tag so the assistant's completion includes that tag.
    """
    marker = "\n<PLAN:"
    idx = text.find(marker)
    if idx == -1:
        return text
    # keep the leading '<PLAN:' (drop the preceding '\n')
    return text[idx + 1:]


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


class Game24SFTChatDataset(Dataset):
    """SFT dataset for Game-24 with chat-template + few-shot prompt.

    Mirrors `Game24SFTDataset` but uses `prompt_builders.fewshot_chat_prompt_24`
    so the SFT baseline trains under the exact same prompt distribution that
    our DAgger stage-2 trains/evals under. The assistant's completion is the
    canonical trajectory starting after "Step 1:" (appended as generation
    prime by the builder).
    """

    def __init__(self, tokenizer, data_path: str, max_seq_len: int = 800,
                  unique_problems: bool = True,
                  prompt_style: str = "fewshot"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        from src.prompt_builders import (
            fewshot_chat_prompt_24, fewshot_chat_prompt_24_plan,
        )
        if prompt_style == "fewshot_plan":
            self._prompt_builder = fewshot_chat_prompt_24_plan
            self._completion_fn = make_completion_plan
        else:
            self._prompt_builder = fewshot_chat_prompt_24
            self._completion_fn = make_completion
        seen = set()
        self.data = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                if unique_problems and item["problem"] in seen:
                    continue
                seen.add(item["problem"])
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text, prompt_add_special = self._prompt_builder(
            self.tokenizer, item["problem"])
        completion = self._completion_fn(item["text"])

        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=prompt_add_special)
        completion_ids = self.tokenizer.encode(
            completion, add_special_tokens=False)
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
