from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerBase

from src.data.dataset import build_prefix_text


def compute_plan_loss(predicted_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    """Continuous proxy for -log p_f(t_i | x, r_<i)."""
    return F.mse_loss(predicted_latent, target_latent)


def compute_reason_loss(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    steps: list[str],
    step_index: int,
    plan_latent: torch.Tensor,
    plan_to_hidden: nn.Module,
    max_question_tokens: int,
    max_step_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Approximate -log p_f(r_i | x, r_<i, t_i) using a virtual embedding token for t_i.
    """
    prefix_text = build_prefix_text(question=question, steps=steps, step_index=step_index)
    target_text = steps[step_index]

    prefix_ids = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_question_tokens,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    target_ids = tokenizer(
        target_text,
        truncation=True,
        max_length=max_step_tokens,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    if target_ids.size(1) == 0:
        target_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

    input_ids = torch.cat([prefix_ids, target_ids], dim=1)
    input_embeds = model.get_input_embeddings()(input_ids)

    prefix_len = prefix_ids.size(1)
    target_len = target_ids.size(1)
    hidden_size = input_embeds.size(-1)

    prefix_embeds = input_embeds[:, :prefix_len, :] if prefix_len > 0 else torch.empty(1, 0, hidden_size, device=device)
    target_embeds = input_embeds[:, prefix_len:, :]
    plan_embed = plan_to_hidden(plan_latent).view(1, 1, -1)

    combined_embeds = torch.cat([prefix_embeds, plan_embed, target_embeds], dim=1)
    seq_len = combined_embeds.size(1)

    labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)
    labels[:, prefix_len + 1 : prefix_len + 1 + target_len] = target_ids
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    outputs = model(
        inputs_embeds=combined_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    return outputs.loss
