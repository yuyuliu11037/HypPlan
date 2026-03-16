from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset

from .preprocessing import encode_sample


def _problem_bucket(problem: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{problem}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_by_problem(
    raw_samples: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in raw_samples:
        grouped[str(sample["problem"])].append(sample)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for problem, group in grouped.items():
        r = _problem_bucket(problem, seed)
        if r < 0.9:
            split = "train"
        elif r < 0.95:
            split = "val"
        else:
            split = "test"
        splits[split].extend(group)
    return splits


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


class PlanningTokenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        split: str,
        max_seq_len: int = 2048,
        seed: int = 42,
        limit: int | None = None,
        presplit: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        raw = load_jsonl(data_path)
        split_records = raw if presplit else split_by_problem(raw, seed=seed)[split]
        if limit is not None:
            random.Random(seed).shuffle(split_records)
            split_records = split_records[:limit]

        encoded: list[dict[str, Any]] = []
        for sample in split_records:
            item = encode_sample(sample, tokenizer=tokenizer, max_seq_len=max_seq_len)
            if item is not None:
                encoded.append(item)
        self.samples = encoded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = 0

    def pad(seq: list[int], fill: int) -> list[int]:
        return seq + [fill] * (max_len - len(seq))

    input_ids = torch.tensor([pad(x["input_ids"], pad_id) for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([pad(x["attention_mask"], 0) for x in batch], dtype=torch.long)
    step_ids = torch.tensor([pad(x["step_ids"], -1) for x in batch], dtype=torch.long)
    labels_stage1 = torch.tensor([pad(x["labels_stage1"], -100) for x in batch], dtype=torch.long)
    labels_stage2 = torch.tensor([pad(x["labels_stage2"], -100) for x in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "step_ids": step_ids,
        "labels_stage1": labels_stage1,
        "labels_stage2": labels_stage2,
        "plan_positions": [x["plan_positions"] for x in batch],
        "step_spans": [x["step_spans"] for x in batch],
        "segment_ids": [x["segment_ids"] for x in batch],
        "within_segment_depths": [x["within_segment_depths"] for x in batch],
        "problems": [x["problem"] for x in batch],
        "ground_truth_answers": [x["ground_truth_answer"] for x in batch],
    }

