from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval import main as eval_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args, _ = parser.parse_known_args()
    sys.argv = [
        sys.argv[0],
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--mode",
        "baseline",
    ]
    eval_main()
