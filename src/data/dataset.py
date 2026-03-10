from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset
import torch

from src.data.schema import DatasetRecord, validate_record

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """Single reasoning path (r_1..r_N) for one question."""

    sample_id: str
    question: str
    steps: List[str]
    answer: str


@dataclass
class AugmentedReasoningPath(ReasoningPath):
    """
    Stores planning latents t_i aligned with reasoning steps r_i.

    `planning_latents[i]` corresponds to `steps[i]`.
    """

    planning_latents: List[torch.Tensor] = field(default_factory=list)


def load_gsm8k_aug_dataset(
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
) -> List[ReasoningPath]:
    hf_dataset = load_dataset(dataset_name, split=split)
    records: List[DatasetRecord] = []
    skipped = 0
    for idx, row in enumerate(hf_dataset):
        if max_samples is not None and len(records) >= max_samples:
            break
        raw = {
            "id": f"{split}-{idx}",
            "question": row.get("question"),
            "steps": row.get("steps"),
            "answer": row.get("answer"),
        }
        try:
            records.append(validate_record(raw, record_idx=idx))
        except ValueError as e:
            skipped += 1
            logger.debug("Skipping invalid record %s: %s", idx, e)
    if skipped:
        logger.info("Skipped %d invalid record(s) in %s/%s", skipped, dataset_name, split)

    if not records:
        raise ValueError(f"No valid records loaded from dataset={dataset_name} split={split}")
    return [
        ReasoningPath(
            sample_id=record.record_id,
            question=record.question,
            steps=record.steps,
            answer=record.answer,
        )
        for record in records
    ]


def initialize_augmented_dataset(
    reasoning_paths: List[ReasoningPath],
    planning_dim: int,
) -> List[AugmentedReasoningPath]:
    augmented: List[AugmentedReasoningPath] = []
    for path in reasoning_paths:
        # Keep cache on CPU so each rank can cheaply sync updates.
        zeros = [torch.zeros(planning_dim, dtype=torch.float32) for _ in path.steps]
        augmented.append(
            AugmentedReasoningPath(
                sample_id=path.sample_id,
                question=path.question,
                steps=path.steps,
                answer=path.answer,
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


def get_rank_shard_indices(total_size: int, process_index: int, num_processes: int) -> List[int]:
    """
    Deterministic strided sharding.

    Example for total=10, world=4:
    rank0: [0,4,8], rank1: [1,5,9], rank2: [2,6], rank3: [3,7]
    """
    if total_size <= 0:
        return []
    if num_processes <= 1:
        return list(range(total_size))
    return list(range(process_index, total_size, num_processes))
