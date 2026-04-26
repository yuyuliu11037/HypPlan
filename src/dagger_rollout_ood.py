"""Generic DAgger rollout for OOD tasks (PQ / BW / GC).

Mirrors src/dagger_rollout_varied.py but uses a TaskAdapter (from
src/dagger_ood_adapters.py) for all task-specific concerns:
  - state representation
  - winning-step oracle
  - step parsing / validation / application
  - state rendering for z input
  - prompt building / step priming
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class StepBoundary:
    step_num: int
    state_before: object
    history_before: tuple
    winning_steps: list  # list of (action, new_state)
    model_action: Optional[object] = None
    model_step_text: Optional[str] = None
    transition_valid: Optional[bool] = None


@dataclass
class Rollout:
    boundaries: list = field(default_factory=list)
    final_state: object = None
    solved: bool = False
    stopped_reason: str = ""
    generation_text: str = ""


@torch.no_grad()
def _compute_z(model, tokenizer, head, up_proj, adapter, state, history,
                device, use_z: bool, random_z: bool = False
                ) -> Optional[torch.Tensor]:
    if not use_z:
        return None
    hidden_dim = up_proj.net[-1].normalized_shape[0]
    if random_z:
        g = torch.randn(1, hidden_dim, device=device)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        g = g * (hidden_dim ** 0.5)
        return g
    state_text = adapter.render_state(state, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True,
                            return_tensors="pt").to(device)
    with model.disable_adapter():
        out = model(input_ids=ids, output_hidden_states=True)
        last_h = out.hidden_states[-1][:, -1, :]
    z_hyp = head(last_h.float())
    return up_proj(z_hyp)


def _sample_next(logits, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sp, si = probs.sort(dim=-1, descending=True)
        cs = sp.cumsum(dim=-1)
        mask = cs - sp > top_p
        sp[mask] = 0.0
        sp /= sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        pick = torch.multinomial(sp, 1)
        return int(si.gather(-1, pick).item())
    return int(torch.multinomial(probs, 1).item())


def _last_step_body(full_gen: str, step_num: int, adapter) -> Optional[str]:
    """Extract the body of step `step_num` from the generation."""
    import re
    pat = re.compile(rf"Step\s+{step_num}:\s*(.*?)(?=\nStep\s+\d+:|Answer\s*:|$)",
                      re.DOTALL)
    m = pat.search(full_gen)
    if m:
        return m.group(1).rstrip().rstrip(".")
    return None


@torch.no_grad()
def rollout_one(model, tokenizer, head, up_proj, adapter, device,
                use_z: bool = True, temperature: float = 0.7,
                top_p: float = 0.95, max_new_tokens: int = 512,
                max_steps: int = 12, random_z: bool = False) -> Rollout:
    """Generate one trajectory for `adapter`. Returns a Rollout with per-
    boundary oracle labels."""
    import re
    result = Rollout()

    prompt_text, add_special = adapter.make_prompt(tokenizer)
    prompt_text = prompt_text + adapter.step_priming_prefix(1)
    input_ids = tokenizer.encode(
        prompt_text, add_special_tokens=add_special, return_tensors="pt",
    ).to(device)

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    generated_ids: list[int] = []
    state = adapter.initial_state
    history: tuple = tuple()

    wins = adapter.winning_steps(state)
    result.boundaries.append(StepBoundary(
        step_num=1, state_before=state, history_before=tuple(),
        winning_steps=wins,
    ))
    if not wins:
        result.stopped_reason = "empty_oracle"
        result.final_state = state
        return result

    pending_z = _compute_z(model, tokenizer, head, up_proj, adapter, state,
                            tuple(), device, use_z=use_z, random_z=random_z)
    prev_count = 0

    BOUNDARY_RE = adapter.BOUNDARY_RE
    TERMINAL_RE = adapter.TERMINAL_RE
    cur_step = 1

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
        cur_count = len(BOUNDARY_RE.findall(full_gen))
        terminal_seen = TERMINAL_RE.search(full_gen) is not None
        if terminal_seen and cur_count <= prev_count:
            cur_count = prev_count + 1

        if cur_count > prev_count:
            prev_count = cur_count
            # Extract just-completed step body. Always prepend the ORIGINAL
            # priming (Step 1:) — the model itself emits Step 2:, Step 3:
            # markers — so the regex finds the right occurrence.
            body = _last_step_body(adapter.step_priming_prefix(1) +
                                     full_gen, cur_step, adapter)
            bdy = result.boundaries[-1]
            if body is None:
                bdy.transition_valid = False
                result.stopped_reason = "parse_fail"
                break
            parsed = adapter.parse_step(body, state, history)
            if parsed is None:
                bdy.transition_valid = False
                result.stopped_reason = "invalid"
                break
            action, raw_text = parsed
            ok, new_state = adapter.validate_apply(state, action)
            bdy.model_action = action
            bdy.model_step_text = raw_text
            bdy.transition_valid = ok
            if not ok:
                result.stopped_reason = "invalid"
                break
            history = history + (action,)
            state = new_state
            cur_step += 1

            if adapter.is_solved(state):
                result.solved = True
                result.stopped_reason = "solved"
                break
            if cur_step > max_steps:
                result.stopped_reason = "budget"
                break

            wins_next = adapter.winning_steps(state)
            result.boundaries.append(StepBoundary(
                step_num=cur_step, state_before=state,
                history_before=history, winning_steps=wins_next,
            ))
            if not wins_next:
                result.stopped_reason = "empty_oracle"
                break
            pending_z = _compute_z(model, tokenizer, head, up_proj, adapter,
                                    state, history, device, use_z=use_z,
                                    random_z=random_z)

    if result.stopped_reason == "":
        result.stopped_reason = "budget"

    result.final_state = state
    result.generation_text = (adapter.step_priming_prefix(1) +
                                tokenizer.decode(generated_ids,
                                                   skip_special_tokens=True
                                                   ).rstrip())
    return result
