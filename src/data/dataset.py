from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch

from src.data.schema import DatasetRecord, validate_record


@dataclass
class ReasoningPath:
    """Single reasoning trajectory (r_1..r_N) for one question."""

    sample_id: str
    source: str
    question: str
    steps: List[str]


@dataclass
class AugmentedReasoningPath(ReasoningPath):
    """
    Stores planning latents t_i aligned with steps r_i.

    `planning_latents[i]` corresponds to step `steps[i]`.
    """

    planning_latents: List[torch.Tensor] = field(default_factory=list)


def is_placeholder_dataset_uri(dataset_uri: str) -> bool:
    return dataset_uri.startswith("TODO://")


def _resolve_jsonl_path(dataset_uri: str) -> Path:
    if dataset_uri.startswith("file://"):
        return Path(dataset_uri[len("file://") :]).expanduser().resolve()
    return Path(dataset_uri).expanduser().resolve()


def flatten_records(records: List[DatasetRecord]) -> List[ReasoningPath]:
    flattened: List[ReasoningPath] = []
    for record in records:
        for trajectory in record.trajectories:
            flattened.append(
                ReasoningPath(
                    sample_id=record.record_id,
                    source=record.source,
                    question=record.question,
                    steps=trajectory.steps,
                )
            )
    return flattened


def load_jsonl_dataset(dataset_uri: str, max_samples: int | None = None) -> List[ReasoningPath]:
    path = _resolve_jsonl_path(dataset_uri)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    records: List[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_samples is not None and len(records) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(validate_record(raw, record_idx=idx))

    if not records:
        raise ValueError(f"No valid records loaded from {path}")

    return flatten_records(records)


def synthetic_reasoning_paths() -> List[ReasoningPath]:
    """
    Small in-memory data for smoke testing when dataset URI is placeholder.
    """
    return [
        ReasoningPath(
            sample_id="synthetic-1",
            source="gsm8k",
            question="If Sara has 3 apples and buys 2 more, how many apples does she have?",
            steps=[
                "Identify initial amount: 3 apples.",
                "Add purchased apples: 3 + 2 = 5.",
                "Answer: Sara has 5 apples.",
            ],
        ),
        ReasoningPath(
            sample_id="synthetic-2",
            source="svamp",
            question="A box has 10 pencils, and 4 are removed. How many remain?",
            steps=[
                "Start with 10 pencils.",
                "Subtract removed pencils: 10 - 4 = 6.",
                "Answer: 6 pencils remain.",
            ],
        ),
    ]


def initialize_augmented_dataset(
    reasoning_paths: List[ReasoningPath],
    planning_dim: int,
    device: torch.device,
) -> List[AugmentedReasoningPath]:
    augmented: List[AugmentedReasoningPath] = []
    for path in reasoning_paths:
        zeros = [torch.zeros(planning_dim, device=device) for _ in path.steps]
        augmented.append(
            AugmentedReasoningPath(
                sample_id=path.sample_id,
                source=path.source,
                question=path.question,
                steps=path.steps,
                planning_latents=zeros,
            )
        )
    return augmented


def build_prefix_text(question: str, steps: List[str], step_index: int) -> str:
    """
    Returns text representing x, r_<i (strict prefix before current step i).
    """
    prior_steps = steps[:step_index]
    if not prior_steps:
        return f"Question:\n{question}\n\nReasoning so far:\n"
    numbered = "\n".join(f"{idx + 1}. {s}" for idx, s in enumerate(prior_steps))
    return f"Question:\n{question}\n\nReasoning so far:\n{numbered}\n"
