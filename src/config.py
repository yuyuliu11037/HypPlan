from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)
