"""DAgger rollout: generate one trajectory for a 24-Game problem and attach
oracle labels at each step boundary.

Mirrors the token-by-token sampling pattern of `src/generate_24_stage2.py`
but with (a) per-step oracle labeling, (b) invalid-state detection and
truncation, (c) a `--use_z` toggle, and (d) an output structure the trainer
can replay.

Separate from training. Rollout is inference-only — no gradients.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import torch

from src.dataset_24 import make_prompt
from src.dataset_24_stage2 import STEP_RE
from src.oracle_24 import (
    apply_step,
    validate_step,
    winning_ops,
)
from src.prompt_builders import sft_prompt_24
from src.tree_data import render_state_from_history


BOUNDARY_RE = re.compile(r"\nStep \d+:")


@dataclass
class StepBoundary:
    """One decision point in a rollout, with oracle labels attached."""
    step_num: int                              # 1-indexed
    problem: str                               # e.g. "4,5,6,10"
    history_before: tuple                      # ((a, op, b, r), ...) up to here
    remaining_before: tuple                    # sorted tuple of Fractions
    winning_ops: list                          # [(op, a, b, r), ...] from oracle
    # Populated after the model emits its step:
    model_parsed: Optional[tuple] = None       # (a, op, b, r) or None
    transition_valid: Optional[bool] = None


@dataclass
class Rollout:
    problem: str
    boundaries: list = field(default_factory=list)
    final_remaining: Optional[tuple] = None
    solved: bool = False
    stopped_reason: str = ""   # solved | invalid | budget | empty_oracle
    generation_text: str = ""  # full model output for debugging


@torch.no_grad()
def _compute_z(model, tokenizer, head, up_proj, problem: str, history: tuple,
               device, use_z: bool, random_z: bool = False) -> Optional[torch.Tensor]:
    """(1, hidden) virtual-token vector, or None when use_z=False.

    When `random_z=True`, return a Gaussian vector rescaled so its L2 norm
    matches what `UpProjector` produces at its LayerNorm output
    (≈ √hidden_dim). This ablates the geometric content of z while
    preserving its injection magnitude.
    """
    if not use_z:
        return None
    hidden_dim = up_proj.net[-1].normalized_shape[0]  # LayerNorm's out_dim
    if random_z:
        g = torch.randn(1, hidden_dim, device=device)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        g = g * (hidden_dim ** 0.5)
        return g
    state_text = render_state_from_history(problem, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True, return_tensors="pt").to(device)
    with model.disable_adapter():
        out = model(input_ids=ids, output_hidden_states=True)
        last_h = out.hidden_states[-1][:, -1, :]
    z_hyp = head(last_h.float())
    return up_proj(z_hyp)  # (1, hidden)


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        csum = sorted_probs.cumsum(dim=-1)
        mask = csum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        pick = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx.gather(-1, pick).item())
    return int(torch.multinomial(probs, 1).item())


_STEP_PREFIX_NOISE = re.compile(r"(Step\s+\d+:)[\s:=]+")


def _parse_history_from_text(step1_text: str) -> tuple:
    """Parse a sequence of 'Step N: a op b = r' from generation text.

    `step1_text` must start with 'Step 1:' (the prompt ended with that and we
    re-prepend for parsing).

    Tolerates common z-injection artifacts by normalizing any combination of
    whitespace / colons / equals immediately after `Step N:` to a single
    space. Handles e.g. `Step 1::`, `Step 1: =`, `Step 1:  ` → `Step 1: `.
    """
    normalized = _STEP_PREFIX_NOISE.sub(r"\1 ", step1_text)
    hist = []
    for m in STEP_RE.finditer(normalized):
        try:
            a = Fraction(m.group(1))
            op = m.group(2)
            b = Fraction(m.group(3))
            r = Fraction(m.group(4).rstrip("."))
        except (ValueError, ZeroDivisionError):
            continue
        hist.append((a, op, b, r))
    return tuple(hist)


@torch.no_grad()
def rollout_one(model, tokenizer, head, up_proj, problem: str, device,
                use_z: bool = True, temperature: float = 0.7, top_p: float = 0.95,
                max_new_tokens: int = 256, max_steps: int = 3,
                prompt_builder=None, random_z: bool = False) -> Rollout:
    """Generate one trajectory with per-step oracle labels.

    `model` must be a PEFT-wrapped causal LM so `model.disable_adapter()`
    works (required to pull frozen-base features for z computation).

    `prompt_builder(tokenizer, problem) -> (text, add_special_tokens)` lets
    callers swap the raw-text prompt for a chat-template few-shot prompt
    (used with Qwen-14B base, see `src.prompt_builders`). Defaults to
    `sft_prompt_24`.
    """
    result = Rollout(problem=problem)

    if prompt_builder is None:
        prompt_builder = sft_prompt_24
    prompt_text, add_special = prompt_builder(tokenizer, problem)
    input_ids = tokenizer.encode(
        prompt_text, add_special_tokens=add_special, return_tensors="pt",
    ).to(device)
    prompt_len = input_ids.size(1)

    # Prime with the prompt.
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    generated_ids: list[int] = []
    embed_table = model.get_input_embeddings()

    initial_remaining = tuple(sorted(Fraction(n)
                                     for n in problem.split(",")))
    history: tuple = tuple()
    remaining: tuple = initial_remaining

    # Record boundary BEFORE step 1 (the one z1 pairs with).
    wins = winning_ops(remaining)
    result.boundaries.append(StepBoundary(
        step_num=1, problem=problem, history_before=tuple(),
        remaining_before=remaining, winning_ops=wins,
    ))
    if not wins:
        result.stopped_reason = "empty_oracle"
        result.final_remaining = remaining
        return result

    # Inject z before the first step's content tokens.
    pending_z = _compute_z(model, tokenizer, head, up_proj, problem,
                            history=tuple(), device=device, use_z=use_z,
                            random_z=random_z)
    prev_boundary_count = 0   # number of "\nStep N:" substrings seen in the
                              # generated-since-prompt text

    for token_idx in range(max_new_tokens):
        if pending_z is not None:
            embeds = pending_z.unsqueeze(1).to(next(model.parameters()).dtype)
            out = model(inputs_embeds=embeds, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            pending_z = None

        next_tok = _sample_next(logits, temperature, top_p)
        if next_tok == tokenizer.eos_token_id:
            break
        generated_ids.append(next_tok)

        # Feed token forward.
        cur = torch.tensor([[next_tok]], device=device)
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        # Boundary detection: count "\nStep N:" matches in the generated text.
        # We prepend "Step 1:" (the prompt ended with it) so STEP_RE can parse
        # the completed earlier step.
        full_gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_with_prefix = "Step 1:" + full_gen
        cur_count = len(BOUNDARY_RE.findall(full_gen))

        if cur_count > prev_boundary_count:
            # A new "\nStep N:" has appeared → the PREVIOUS step is now complete.
            prev_boundary_count = cur_count
            parsed = _parse_history_from_text(full_with_prefix)
            just_finished_idx = len(history)
            if len(parsed) <= just_finished_idx:
                # Parse failure on the step that should have just completed.
                result.boundaries[-1].transition_valid = False
                result.stopped_reason = "invalid"
                break
            step_tuple = parsed[just_finished_idx]
            a, op, b, r = step_tuple
            ok, _ = validate_step(remaining, a, op, b, r)
            bdy = result.boundaries[-1]
            bdy.model_parsed = step_tuple
            bdy.transition_valid = ok
            if not ok:
                result.stopped_reason = "invalid"
                break
            history = history + (step_tuple,)
            remaining = apply_step(remaining, a, b, r)

            # Are we at the last step?
            if len(remaining) == 1:
                result.solved = (remaining[0] == Fraction(24))
                result.stopped_reason = "solved" if result.solved else "invalid"
                break
            if len(history) >= max_steps:
                result.stopped_reason = "budget"
                break

            # Record the NEXT boundary and schedule z for it.
            wins_next = winning_ops(remaining)
            result.boundaries.append(StepBoundary(
                step_num=len(history) + 1, problem=problem,
                history_before=history, remaining_before=remaining,
                winning_ops=wins_next,
            ))
            if not wins_next:
                result.stopped_reason = "empty_oracle"
                break
            pending_z = _compute_z(model, tokenizer, head, up_proj, problem,
                                     history=history, device=device, use_z=use_z,
                                     random_z=random_z)

    # If we exited without setting a reason (e.g. max_new_tokens cutoff mid-step),
    # mark as budget-exhausted. The last boundary's transition_valid is None.
    if result.stopped_reason == "":
        result.stopped_reason = "budget"

    result.final_remaining = remaining
    result.generation_text = "Step 1:" + tokenizer.decode(
        generated_ids, skip_special_tokens=True).rstrip()
    return result
