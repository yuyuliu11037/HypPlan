from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.data.schema import PlanningSample, Span, TokenizedPlanningSample


def build_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nReasoning:\n"


def build_step_text(step_index: int, step_text: str) -> str:
    return f"Step {step_index}: {step_text.strip()}\n"


def build_answer_text(answer: str, add_eos_marker: bool = False) -> str:
    text = f"Final Answer: {answer.strip()}"
    if add_eos_marker:
        text = f"{text}\n"
    return text


def normalize_steps(steps: Any) -> list[str]:
    if isinstance(steps, list):
        normalized = [str(step).strip() for step in steps if str(step).strip()]
        if normalized:
            return normalized

    if isinstance(steps, str) and steps.strip():
        return [line.strip() for line in steps.splitlines() if line.strip()]

    raise ValueError(f"Unsupported steps value: {steps!r}")


def format_sample(question: str, steps: Any, answer: str) -> PlanningSample:
    normalized_steps = normalize_steps(steps)
    prompt_text = build_prompt(question)
    step_texts = [build_step_text(index + 1, step) for index, step in enumerate(normalized_steps)]
    answer_text = build_answer_text(answer)
    return PlanningSample(
        question=question.strip(),
        steps=normalized_steps,
        answer=answer.strip(),
        prompt_text=prompt_text,
        step_texts=step_texts,
        answer_text=answer_text,
    )


def tokenize_formatted_sample(
    sample: PlanningSample,
    tokenizer: PreTrainedTokenizerBase,
    add_eos_token: bool = True,
) -> TokenizedPlanningSample:
    prompt_ids = tokenizer(sample.prompt_text, add_special_tokens=False)["input_ids"]
    step_ids = [
        tokenizer(step_text, add_special_tokens=False)["input_ids"]
        for step_text in sample.step_texts
    ]
    answer_ids = tokenizer(sample.answer_text, add_special_tokens=False)["input_ids"]

    if add_eos_token and tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = list(prompt_ids)
    step_spans: list[Span] = []
    cursor = len(prompt_ids)

    for token_ids in step_ids:
        input_ids.extend(token_ids)
        step_spans.append(Span(start=cursor, end=cursor + len(token_ids)))
        cursor += len(token_ids)

    answer_span = Span(start=cursor, end=cursor + len(answer_ids))
    input_ids.extend(answer_ids)
    attention_mask = [1] * len(input_ids)

    return TokenizedPlanningSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_length=len(prompt_ids),
        step_spans=step_spans,
        answer_span=answer_span,
        question=sample.question,
        steps=sample.steps,
        answer=sample.answer,
        prompt_text=sample.prompt_text,
        step_texts=sample.step_texts,
        answer_text=sample.answer_text,
    )


class GSM8KPlanningDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        add_eos_token: bool = True,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos_token = add_eos_token

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.hf_dataset[index]
        formatted = format_sample(row["question"], row["steps"], row["answer"])
        tokenized = tokenize_formatted_sample(
            formatted,
            tokenizer=self.tokenizer,
            add_eos_token=self.add_eos_token,
        )

        if tokenized.sequence_length > self.max_length:
            raise ValueError(
                f"Sample {index} length {tokenized.sequence_length} exceeds max_length={self.max_length}."
            )

        return asdict(tokenized)


class PlanningCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            sequence_length = len(item["input_ids"])
            pad_length = max_length - sequence_length

            padded_ids = item["input_ids"] + [self.pad_token_id] * pad_length
            padded_mask = item["attention_mask"] + [0] * pad_length
            padded_labels = [-100] * item["prompt_length"] + item["input_ids"][item["prompt_length"] :]
            padded_labels = padded_labels + [-100] * pad_length

            input_ids.append(padded_ids)
            attention_mask.append(padded_mask)
            labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_lengths": [item["prompt_length"] for item in batch],
            "step_spans": [
                [Span(**span) if isinstance(span, dict) else span for span in item["step_spans"]]
                for item in batch
            ],
            "answer_spans": [
                Span(**item["answer_span"]) if isinstance(item["answer_span"], dict) else item["answer_span"]
                for item in batch
            ],
            "questions": [item["question"] for item in batch],
            "steps": [item["steps"] for item in batch],
            "answers": [item["answer"] for item in batch],
            "prompt_texts": [item["prompt_text"] for item in batch],
            "step_texts": [item["step_texts"] for item in batch],
            "answer_texts": [item["answer_text"] for item in batch],
        }


def load_gsm8k_aug_splits(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["data"]
    dataset_name = dataset_cfg.get("dataset_name", "whyNLP/gsm8k-aug")
    dataset = load_dataset(dataset_name)

    if "validation" in dataset:
        return dataset

    train_split = dataset[dataset_cfg.get("train_split", "train")]
    validation_size = dataset_cfg.get("validation_size", 0.01)
    seed = dataset_cfg.get("seed", 42)

    if isinstance(validation_size, float):
        validation_size = max(1, math.floor(len(train_split) * validation_size))

    split = train_split.train_test_split(test_size=validation_size, seed=seed, shuffle=True)
    return DatasetDict(train=split["train"], validation=split["test"])


def build_dataloaders(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    training_cfg = config["training"]
    dataset = load_gsm8k_aug_splits(config)

    train_dataset = GSM8KPlanningDataset(
        dataset["train"],
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
        add_eos_token=data_cfg.get("add_eos_token", True),
    )
    val_dataset = GSM8KPlanningDataset(
        dataset["validation"],
        tokenizer=tokenizer,
        max_length=data_cfg["max_length"],
        add_eos_token=data_cfg.get("add_eos_token", True),
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    collator = PlanningCollator(pad_token_id=pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get("eval_batch_size", training_cfg["batch_size"]),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def sample_validation_examples(dataset, count: int, seed: int = 42) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return [dataset[index] for index in indices[:count]]
