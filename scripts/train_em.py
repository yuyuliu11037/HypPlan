from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.training.em_trainer import EMTrainer, TrainConfig


def load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Near-real EM trainer for planning tokens.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = load_yaml(args.config)
    cfg = TrainConfig.from_dict(cfg_dict)
    trainer = EMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
