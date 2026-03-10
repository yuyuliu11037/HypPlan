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
    max_question_tokens: int,
    max_step_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    return compute_reason_loss_batch(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        steps_batch=[steps],
        step_indices=[step_index],
        plan_latents=plan_latent.view(1, -1),
        max_question_tokens=max_question_tokens,
        max_step_tokens=max_step_tokens,
        device=device,
    )


def compute_reason_loss_batch(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    steps_batch: list[list[str]],
    step_indices: list[int],
    plan_latents: torch.Tensor,
    max_question_tokens: int,
    max_step_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Batched approximation to -log p_f(r_i | x, r_<i, t_i) using a virtual embedding token for t_i.
    """
    if not questions:
        raise ValueError("questions must be non-empty")
    batch_size = len(questions)
    if len(steps_batch) != batch_size or len(step_indices) != batch_size:
        raise ValueError("Batch inputs must share the same length")
    if plan_latents.dim() != 2 or plan_latents.size(0) != batch_size:
        raise ValueError("plan_latents must have shape [batch_size, hidden_size]")

    prefix_texts = [
        build_prefix_text(question=q, steps=s, step_index=i)
        for q, s, i in zip(questions, steps_batch, step_indices)
    ]
    target_texts = [steps[i] for steps, i in zip(steps_batch, step_indices)]

    prefix_tok = tokenizer(
        prefix_texts,
        truncation=True,
        max_length=max_question_tokens,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    target_tok = tokenizer(
        target_texts,
        truncation=True,
        max_length=max_step_tokens,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    prefix_ids = prefix_tok.input_ids.to(device)
    prefix_mask = prefix_tok.attention_mask.to(device)
    target_ids = target_tok.input_ids.to(device)
    target_mask = target_tok.attention_mask.to(device)

    # Keep at least one supervised token per example.
    empty_targets = target_mask.sum(dim=1) == 0
    if torch.any(empty_targets):
        target_ids[empty_targets, 0] = tokenizer.eos_token_id
        target_mask[empty_targets, 0] = 1

    input_ids = torch.cat([prefix_ids, target_ids], dim=1)
    # Unwrap DDP so we can call get_input_embeddings (wrapper does not expose it)
    model_for_embeds = getattr(model, "module", model)
    input_embeds = model_for_embeds.get_input_embeddings()(input_ids)

    prefix_pad_len = prefix_ids.size(1)
    target_pad_len = target_ids.size(1)
    hidden_size = input_embeds.size(-1)
    seq_len = prefix_pad_len + 1 + target_pad_len

    combined_embeds = torch.zeros((batch_size, seq_len, hidden_size), dtype=input_embeds.dtype, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    plan_embed = plan_latents.to(device=device, dtype=input_embeds.dtype)
    target_embeds = input_embeds[:, prefix_pad_len:, :]

    for b in range(batch_size):
        prefix_len = int(prefix_mask[b].sum().item())
        target_len = int(target_mask[b].sum().item())

        if prefix_len > 0:
            combined_embeds[b, :prefix_len, :] = input_embeds[b, :prefix_len, :]
        combined_embeds[b, prefix_len, :] = plan_embed[b]
        combined_embeds[b, prefix_len + 1 : prefix_len + 1 + target_len, :] = target_embeds[b, :target_len, :]

        attention_mask[b, : prefix_len + 1 + target_len] = 1
        labels[b, prefix_len + 1 : prefix_len + 1 + target_len] = target_ids[b, :target_len]

    outputs = model(
        inputs_embeds=combined_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    return outputs.loss
