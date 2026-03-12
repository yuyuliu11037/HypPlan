from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.model.qwen_planning import QwenPlanningModel
from src.training.distributed import silence_non_zero_local_ranks
from src.training.trainer import ExperimentTrainer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CoT-SFT baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    silence_non_zero_local_ranks()
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = build_dataloaders(config, tokenizer)
    model = QwenPlanningModel(config)

    output_dir = Path(config["output_dir"])
    trainer = ExperimentTrainer(model=model, tokenizer=tokenizer, config=config, output_dir=output_dir)
    trainer.train_baseline(train_loader, val_loader)


if __name__ == "__main__":
    main()
