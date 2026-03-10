from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root so "src" is importable when running this script directly.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        print("Running single-process training. Use `accelerate launch --num_processes 4` to use 4 GPUs.")
    else:
        print(f"Running distributed training: WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    cfg = TrainConfig.from_dict(cfg_dict)
    trainer = EMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
