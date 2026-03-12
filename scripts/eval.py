from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.model.qwen_planning import QwenPlanningModel
from src.training.distributed import silence_non_zero_local_ranks
from src.training.eval import evaluate_generation, evaluate_loss
from src.training.trainer import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate planning or baseline checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path saved by training.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["planning", "baseline"],
        required=True,
        help="Evaluation mode.",
    )
    return parser.parse_args()


def main() -> None:
    silence_non_zero_local_ranks()
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    distributed_cfg = config.get("distributed", {})
    accelerator = Accelerator(mixed_precision=distributed_cfg.get("mixed_precision", "no"))

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, val_loader = build_dataloaders(config, tokenizer)
    model = QwenPlanningModel(config)

    if args.mode == "baseline" and config["model"].get("use_lora_for_baseline", True):
        model.enable_lora(config["model"].get("lora", {}))
    if args.mode == "planning" and config["model"].get("use_lora", True):
        model.enable_lora(config["model"].get("lora", {}))

    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_trainable_state_dict(state["model_state"])
    model, val_loader = accelerator.prepare(model, val_loader)
    raw_model = accelerator.unwrap_model(model)

    if args.mode == "planning":
        loss_metrics = evaluate_loss(raw_model, val_loader, mode="planning_stage2", accelerator=accelerator)
    else:
        loss_metrics = evaluate_loss(raw_model, val_loader, mode="baseline", accelerator=accelerator)

    generation_metrics = evaluate_generation(
        model=raw_model,
        tokenizer=tokenizer,
        dataloader=val_loader,
        mode=args.mode,
        max_examples=config.get("generation", {}).get("eval_examples", 32),
        generation_cfg=config.get("generation", {}),
        accelerator=accelerator,
    )

    accelerator.print({"loss": loss_metrics, "generation": generation_metrics})


if __name__ == "__main__":
    main()
