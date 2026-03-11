from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PlanningSample:
    question: str
    steps: list[str]
    answer: str
    prompt_text: str
    step_texts: list[str]
    answer_text: str


@dataclass(frozen=True)
class TokenizedPlanningSample:
    input_ids: list[int]
    attention_mask: list[int]
    prompt_length: int
    step_spans: list[Span]
    answer_span: Span
    question: str
    steps: list[str]
    answer: str
    prompt_text: str
    step_texts: list[str]
    answer_text: str

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)
