from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import build_training_example


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _split_by_problem(
    rows: list[dict[str, Any]],
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    if rows and "split" in rows[0]:
        return [r for r in rows if r.get("split") == split]

    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(row["problem"], []).append(row)

    problems = list(by_problem.keys())
    rng = random.Random(seed)
    rng.shuffle(problems)

    n = len(problems)
    n_train = int(0.90 * n)
    n_val = int(0.05 * n)

    if split == "train":
        selected = set(problems[:n_train])
    elif split == "val":
        selected = set(problems[n_train : n_train + n_val])
    else:
        selected = set(problems[n_train + n_val :])

    out: list[dict[str, Any]] = []
    for p in selected:
        out.extend(by_problem[p])
    return out


class PlanningTokenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        split: str = "train",
        seed: int = 42,
        max_seq_len: int = 2048,
        plan_token: str = "[PLAN]",
        max_segments: int = 16,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_segments = max_segments

        plan_token_id = tokenizer.convert_tokens_to_ids(plan_token)
        if plan_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Token {plan_token!r} is not in tokenizer vocab. "
                "Call tokenizer.add_special_tokens first."
            )

        rows = _read_jsonl(data_path)
        rows = [r for r in rows if len(r.get("steps", [])) > 1]
        rows = _split_by_problem(rows, split=split, seed=seed)

        samples: list[dict[str, Any]] = []
        for sample_idx, row in enumerate(rows):
            ex = build_training_example(
                problem=row["problem"],
                steps=row["steps"],
                tokenizer=tokenizer,
                plan_token_id=plan_token_id,
                max_seq_len=max_seq_len,
                max_segments=max_segments,
            )
            if ex is None:
                continue
            ex["solution_uid"] = sample_idx
            ex["problem"] = row["problem"]
            ex["ground_truth_answer"] = row.get("ground_truth_answer", "")
            samples.append(ex)

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Batch is empty.")

    bs = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)
    max_plan = max(len(x["plan_positions"]) for x in batch)

    input_ids = torch.full((bs, max_len), fill_value=0, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    step_ids = torch.full((bs, max_len), fill_value=-1, dtype=torch.long)
    token_loss_mask = torch.zeros((bs, max_len), dtype=torch.bool)

    plan_positions = torch.full((bs, max_plan), fill_value=-1, dtype=torch.long)
    plan_mask = torch.zeros((bs, max_plan), dtype=torch.bool)
    segment_ids_raw = torch.full((bs, max_plan), fill_value=-1, dtype=torch.long)
    segment_labels = torch.full((bs, max_plan), fill_value=-1, dtype=torch.long)
    depth_labels = torch.zeros((bs, max_plan), dtype=torch.float)
    solution_ids = torch.full((bs, max_plan), fill_value=-1, dtype=torch.long)

    for i, ex in enumerate(batch):
        seq_len = len(ex["input_ids"])
        n_plan = len(ex["plan_positions"])

        input_ids[i, :seq_len] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        step_ids[i, :seq_len] = torch.tensor(ex["step_ids"], dtype=torch.long)
        token_loss_mask[i, :seq_len] = torch.tensor(ex["token_loss_mask"], dtype=torch.bool)

        if n_plan > 0:
            plan_positions[i, :n_plan] = torch.tensor(ex["plan_positions"], dtype=torch.long)
            plan_mask[i, :n_plan] = True
            segment_ids_raw[i, :n_plan] = torch.tensor(ex["segment_ids_raw"], dtype=torch.long)
            segment_labels[i, :n_plan] = torch.tensor(ex["segment_labels"], dtype=torch.long)
            depth_labels[i, :n_plan] = torch.tensor(ex["depth_labels"], dtype=torch.float)
            solution_ids[i, :n_plan] = ex["solution_uid"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "step_ids": step_ids,
        "token_loss_mask": token_loss_mask,
        "plan_positions": plan_positions,
        "plan_mask": plan_mask,
        "segment_ids_raw": segment_ids_raw,
        "segment_labels": segment_labels,
        "depth_labels": depth_labels,
        "solution_ids": solution_ids,
    }
