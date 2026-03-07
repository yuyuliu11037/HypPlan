from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: List[str]


class PlanningHead(nn.Module):
    """Small MLP mapping decoder hidden state to planning latent."""

    def __init__(self, hidden_size: int, planning_dim: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, planning_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)


def load_tokenizer_and_model(
    model_name_or_path: str,
    lora_settings: LoraSettings,
    device_map: str | None = None,
) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Load causal LM and attach LoRA adapters.

    Returns tokenizer and PEFT-wrapped model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_settings.rank,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        target_modules=lora_settings.target_modules,
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model
