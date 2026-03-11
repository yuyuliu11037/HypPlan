from __future__ import annotations

import re
from math import ceil
from typing import Any

import torch
from tqdm.auto import tqdm

from src.data.dataset import build_prompt


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_answer_candidate(text: str) -> str:
    if "Final Answer:" in text:
        return text.rsplit("Final Answer:", maxsplit=1)[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def numeric_match(prediction: str, reference: str) -> bool:
    pred_num = extract_last_number(extract_answer_candidate(prediction))
    ref_num = extract_last_number(reference)
    return pred_num is not None and pred_num == ref_num


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer_text(extract_answer_candidate(prediction)) == normalize_answer_text(reference)


def evaluate_loss(
    model,
    dataloader,
    mode: str,
    max_batches: int | None = None,
    accelerator=None,
) -> dict[str, float]:
    model.eval()
    device = accelerator.device if accelerator is not None else model.device
    total_weighted_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device, dtype=torch.long)

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            if mode == "planning_stage1":
                metrics = model.compute_stage1_loss(batch)
            elif mode == "planning_stage2":
                metrics = model.compute_stage2_loss(batch)
            elif mode == "baseline":
                metrics = model.compute_baseline_loss(batch)
            else:
                raise ValueError(f"Unsupported eval mode: {mode}")

            total_weighted_loss = total_weighted_loss + (metrics["loss"] * metrics["token_count"])
            total_tokens = total_tokens + metrics["token_count"]

    model.train()
    if accelerator is not None:
        total_weighted_loss = accelerator.reduce(total_weighted_loss, reduction="sum")
        total_tokens = accelerator.reduce(total_tokens, reduction="sum")

    return {"loss": float((total_weighted_loss / total_tokens.clamp_min(1)).item())}


@torch.no_grad()
def generate_planning_prediction(
    model,
    tokenizer,
    question: str,
    step_count: int,
    max_step_tokens: int,
    max_answer_tokens: int,
    temperature: float = 0.0,
) -> str:
    prompt = build_prompt(question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
    newline_tokens = tokenizer("\n", add_special_tokens=False)["input_ids"]
    newline_token_id = newline_tokens[0] if len(newline_tokens) == 1 else None

    context_ids = prompt_ids
    generated_steps: list[str] = []

    for _ in range(step_count):
        step_ids = model.generate_step(
            context_ids=context_ids,
            max_new_tokens=max_step_tokens,
            stop_token_id=newline_token_id,
            temperature=temperature,
        )
        if step_ids.numel() == 0:
            break
        context_ids = torch.cat([context_ids, step_ids], dim=0)
        generated_steps.append(tokenizer.decode(step_ids, skip_special_tokens=True))

    answer_ids = model.generate_answer(
        context_ids=context_ids,
        max_new_tokens=max_answer_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
    )
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return "".join(generated_steps) + answer_text


@torch.no_grad()
def generate_baseline_prediction(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    attention_mask = torch.ones_like(input_ids)

    if temperature and temperature > 0:
        generated = model.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        generated = model.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    continuation = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True)


def evaluate_generation(
    model,
    tokenizer,
    dataloader,
    mode: str,
    max_examples: int,
    generation_cfg: dict[str, Any],
    accelerator=None,
) -> dict[str, float]:
    if max_examples <= 0:
        return {"exact_match": 0.0, "numeric_match": 0.0}

    model.eval()
    world_size = accelerator.num_processes if accelerator is not None else 1
    local_budget = ceil(max_examples / world_size)

    exact_hits = torch.zeros((), device=model.device)
    numeric_hits = torch.zeros((), device=model.device)
    seen = torch.zeros((), device=model.device)

    iterator = tqdm(
        dataloader,
        desc="generation-eval",
        leave=False,
        disable=accelerator is not None and not accelerator.is_local_main_process,
    )

    for batch in iterator:
        batch_size = len(batch["questions"])
        for batch_index in range(batch_size):
            if int(seen.item()) >= local_budget:
                break

            question = batch["questions"][batch_index]
            reference = batch["answers"][batch_index]

            if mode == "planning":
                prediction = generate_planning_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    step_count=len(batch["step_spans"][batch_index]),
                    max_step_tokens=generation_cfg.get("max_step_tokens", 32),
                    max_answer_tokens=generation_cfg.get("max_answer_tokens", 32),
                    temperature=generation_cfg.get("temperature", 0.0),
                )
            elif mode == "baseline":
                prediction = generate_baseline_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=batch["prompt_texts"][batch_index],
                    max_new_tokens=generation_cfg.get("max_new_tokens", 128),
                    temperature=generation_cfg.get("temperature", 0.0),
                )
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")

            exact_hits += int(exact_match(prediction, reference))
            numeric_hits += int(numeric_match(prediction, reference))
            seen += 1

        if int(seen.item()) >= local_budget:
            break

    model.train()
    if accelerator is not None:
        exact_hits = accelerator.reduce(exact_hits, reduction="sum")
        numeric_hits = accelerator.reduce(numeric_hits, reduction="sum")
        seen = accelerator.reduce(seen, reduction="sum")

    if int(seen.item()) == 0:
        return {"exact_match": 0.0, "numeric_match": 0.0}
    return {
        "exact_match": float((exact_hits / seen).item()),
        "numeric_match": float((numeric_hits / seen).item()),
    }
