from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


@dataclass
class EvalRow:
    sample_id: str
    question: str
    reference: str
    prediction: str
    exact_match: bool
    reference_substring_match: bool
    last_number_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "question": self.question,
            "reference": self.reference,
            "prediction": self.prediction,
            "exact_match": self.exact_match,
            "reference_substring_match": self.reference_substring_match,
            "last_number_match": self.last_number_match,
        }


def score_prediction(reference: str, prediction: str) -> Dict[str, bool]:
    ref_norm = normalize_text(reference)
    pred_norm = normalize_text(prediction)

    exact_match = ref_norm == pred_norm
    reference_substring_match = ref_norm in pred_norm

    ref_num = extract_last_number(reference)
    pred_num = extract_last_number(prediction)
    last_number_match = ref_num is not None and pred_num is not None and ref_num == pred_num

    return {
        "exact_match": exact_match,
        "reference_substring_match": reference_substring_match,
        "last_number_match": last_number_match,
    }


def write_eval_results(rows: List[EvalRow], output_path: Path) -> Dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    total = len(rows)
    if total == 0:
        return {
            "count": 0.0,
            "exact_match": 0.0,
            "reference_substring_match": 0.0,
            "last_number_match": 0.0,
        }

    exact = sum(1 for r in rows if r.exact_match) / total
    substring = sum(1 for r in rows if r.reference_substring_match) / total
    number = sum(1 for r in rows if r.last_number_match) / total
    return {
        "count": float(total),
        "exact_match": exact,
        "reference_substring_match": substring,
        "last_number_match": number,
    }
