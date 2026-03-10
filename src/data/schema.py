from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DatasetRecord:
    record_id: str
    question: str
    steps: List[str]
    answer: str


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_steps(value: Any, record_idx: int) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Record {record_idx}: 'steps' must be a non-empty list of strings."
        )
    cleaned_steps: List[str] = []
    for step_idx, step in enumerate(value):
        if not _is_non_empty_string(step):
            raise ValueError(
                f"Record {record_idx} step {step_idx}: each step must be a non-empty string."
            )
        cleaned_steps.append(step.strip())
    return cleaned_steps


def validate_record(raw: Dict[str, Any], record_idx: int) -> DatasetRecord:
    if not _is_non_empty_string(raw.get("question")):
        raise ValueError(f"Record {record_idx}: missing or invalid 'question'.")
    if not _is_non_empty_string(raw.get("answer")):
        raise ValueError(f"Record {record_idx}: missing or invalid 'answer'.")
    steps = _validate_steps(raw.get("steps"), record_idx)
    raw_id = raw.get("id")
    record_id = str(raw_id).strip() if _is_non_empty_string(raw_id) else f"gsm8k-aug-{record_idx}"

    return DatasetRecord(
        record_id=record_id,
        question=str(raw["question"]).strip(),
        steps=steps,
        answer=str(raw["answer"]).strip(),
    )
