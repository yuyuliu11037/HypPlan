"""DAgger rollout for varied-target Game-of-24.

Fork of [src/dagger_rollout.py](src/dagger_rollout.py):
  - takes (pool, target) instead of a fixed "a,b,c,d" problem string,
  - uses the target-parameterized oracle and generic state renderer,
  - checks solved against the per-problem target (not fixed 24).

All other mechanics (boundary detection, z-injection, invalid-step truncation,
--use_z toggle, --random_z ablation) mirror the original.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import torch

from src.dataset_24_stage2 import STEP_RE
from src.oracle_24_varied import apply_step, validate_step, winning_ops
from src.prompt_builders import fewshot_chat_prompt_generic
from src.tree_data_generic import render_state_generic


BOUNDARY_RE = re.compile(r"\nStep \d+:")
_STEP_PREFIX_NOISE = re.compile(r"(Step\s+\d+:)[\s:=]+")


@dataclass
class StepBoundary:
    step_num: int
    pool: list
    target: int
    history_before: tuple
    remaining_before: tuple
    winning_ops: list
    model_parsed: Optional[tuple] = None
    transition_valid: Optional[bool] = None


@dataclass
class Rollout:
    pool: list
    target: int
    boundaries: list = field(default_factory=list)
    final_remaining: Optional[tuple] = None
    solved: bool = False
    stopped_reason: str = ""
    generation_text: str = ""


@torch.no_grad()
def _compute_z(model, tokenizer, head, up_proj, pool: list, target: int,
               history: tuple, device, use_z: bool, random_z: bool = False
               ) -> Optional[torch.Tensor]:
    """(1, hidden) virtual-token vector, or None when use_z=False."""
    if not use_z:
        return None
    hidden_dim = up_proj.net[-1].normalized_shape[0]
    if random_z:
        g = torch.randn(1, hidden_dim, device=device)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        g = g * (hidden_dim ** 0.5)
        return g
    state_text = render_state_generic(pool, target, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True,
                           return_tensors="pt").to(device)
    with model.disable_adapter():
        out = model(input_ids=ids, output_hidden_states=True)
        last_h = out.hidden_states[-1][:, -1, :]
    z_hyp = head(last_h.float())
    return up_proj(z_hyp)


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


def _parse_history_from_text(step1_text: str) -> tuple:
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
def rollout_one(model, tokenizer, head, up_proj, pool: list, target: int,
                device, use_z: bool = True, temperature: float = 0.7,
                top_p: float = 0.95, max_new_tokens: int = 256,
                max_steps: int = 3, random_z: bool = False) -> Rollout:
    """Generate one trajectory for (pool, target). Returns a Rollout with
    per-boundary oracle labels."""
    pool = list(pool)
    result = Rollout(pool=pool, target=int(target))

    # Trivial 0-step case: pool already equals [target]. Mark solved and
    # skip generation — DAgger has nothing to learn here.
    if len(pool) == 1:
        result.solved = (Fraction(int(pool[0])) == Fraction(int(target)))
        result.stopped_reason = "solved" if result.solved else "invalid"
        result.final_remaining = tuple(Fraction(int(x)) for x in pool)
        return result

    # Generic fewshot prompt; append "Step 1:" to prime rollout parsing
    # identically to fixed-target. Trivial cases were handled above.
    prompt_text, add_special = fewshot_chat_prompt_generic(tokenizer, pool,
                                                            target)
    prompt_text = prompt_text + "Step 1:"

    input_ids = tokenizer.encode(
        prompt_text, add_special_tokens=add_special, return_tensors="pt",
    ).to(device)

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    generated_ids: list[int] = []

    initial_remaining = tuple(sorted(Fraction(int(n)) for n in pool))
    history: tuple = tuple()
    remaining: tuple = initial_remaining

    # Boundary before step 1
    wins = winning_ops(remaining, target)
    result.boundaries.append(StepBoundary(
        step_num=1, pool=pool, target=int(target), history_before=tuple(),
        remaining_before=remaining, winning_ops=wins,
    ))
    if not wins:
        result.stopped_reason = "empty_oracle"
        result.final_remaining = remaining
        return result

    pending_z = _compute_z(model, tokenizer, head, up_proj, pool, target,
                            history=tuple(), device=device, use_z=use_z,
                            random_z=random_z)
    prev_boundary_count = 0

    for token_idx in range(max_new_tokens):
        if pending_z is not None:
            embeds = pending_z.unsqueeze(1).to(
                next(model.parameters()).dtype)
            out = model(inputs_embeds=embeds, past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            pending_z = None

        next_tok = _sample_next(logits, temperature, top_p)
        if next_tok == tokenizer.eos_token_id:
            break
        generated_ids.append(next_tok)

        cur = torch.tensor([[next_tok]], device=device)
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        full_gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_with_prefix = "Step 1:" + full_gen
        cur_count = len(BOUNDARY_RE.findall(full_gen))

        if cur_count > prev_boundary_count:
            prev_boundary_count = cur_count
            parsed = _parse_history_from_text(full_with_prefix)
            just_finished_idx = len(history)
            if len(parsed) <= just_finished_idx:
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

            if len(remaining) == 1:
                result.solved = (remaining[0] == Fraction(int(target)))
                result.stopped_reason = "solved" if result.solved else "invalid"
                break
            if len(history) >= max_steps:
                result.stopped_reason = "budget"
                break

            wins_next = winning_ops(remaining, target)
            result.boundaries.append(StepBoundary(
                step_num=len(history) + 1, pool=pool, target=int(target),
                history_before=history, remaining_before=remaining,
                winning_ops=wins_next,
            ))
            if not wins_next:
                result.stopped_reason = "empty_oracle"
                break
            pending_z = _compute_z(model, tokenizer, head, up_proj,
                                    pool, target, history=history,
                                    device=device, use_z=use_z,
                                    random_z=random_z)

    if result.stopped_reason == "":
        result.stopped_reason = "budget"

    result.final_remaining = remaining
    result.generation_text = "Step 1:" + tokenizer.decode(
        generated_ids, skip_special_tokens=True).rstrip()
    return result
