from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal


AllowedSource = Literal["gsm8k", "math", "aqua", "svamp"]
ALLOWED_SOURCES = {"gsm8k", "math", "aqua", "svamp"}


@dataclass
class Trajectory:
    steps: List[str]


@dataclass
class DatasetRecord:
    record_id: str
    source: AllowedSource
    question: str
    trajectories: List[Trajectory]


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_steps(value: Any, record_idx: int, traj_idx: int) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Record {record_idx} trajectory {traj_idx}: 'steps' must be a non-empty list of strings."
        )
    cleaned_steps: List[str] = []
    for step_idx, step in enumerate(value):
        if not _is_non_empty_string(step):
            raise ValueError(
                f"Record {record_idx} trajectory {traj_idx} step {step_idx}: each step must be a non-empty string."
            )
        cleaned_steps.append(step.strip())
    return cleaned_steps


def validate_record(raw: Dict[str, Any], record_idx: int) -> DatasetRecord:
    if not _is_non_empty_string(raw.get("id")):
        raise ValueError(f"Record {record_idx}: missing or invalid 'id'.")
    if raw.get("source") not in ALLOWED_SOURCES:
        raise ValueError(
            f"Record {record_idx}: 'source' must be one of {sorted(ALLOWED_SOURCES)}."
        )
    if not _is_non_empty_string(raw.get("question")):
        raise ValueError(f"Record {record_idx}: missing or invalid 'question'.")
    if not isinstance(raw.get("trajectories"), list) or not raw["trajectories"]:
        raise ValueError(
            f"Record {record_idx}: 'trajectories' must be a non-empty list."
        )

    trajectories: List[Trajectory] = []
    for traj_idx, traj in enumerate(raw["trajectories"]):
        if not isinstance(traj, dict):
            raise ValueError(
                f"Record {record_idx} trajectory {traj_idx}: expected object with 'steps'."
            )
        steps = _validate_steps(traj.get("steps"), record_idx, traj_idx)
        trajectories.append(Trajectory(steps=steps))

    return DatasetRecord(
        record_id=str(raw["id"]).strip(),
        source=raw["source"],
        question=str(raw["question"]).strip(),
        trajectories=trajectories,
    )
