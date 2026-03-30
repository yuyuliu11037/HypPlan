"""LoRA setup utilities using peft."""
from __future__ import annotations

from peft import LoraConfig, get_peft_model, TaskType


def setup_lora(model, lora_r: int, lora_alpha: int, target_modules: list[str]):
    """Apply LoRA adapters to a causal LM model.

    Args:
        model: HuggingFace causal LM model.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        target_modules: List of module names to apply LoRA to.
    Returns:
        PeftModel with LoRA adapters.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)
