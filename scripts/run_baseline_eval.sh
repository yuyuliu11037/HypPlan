#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# For a quick eval, add: --subset_ratio 0.1
torchrun --nproc_per_node=4 src/evaluation/baseline_eval.py \
  --model_name Qwen/Qwen2.5-7B \
  --max_new_tokens 1024 \
  --output_file results/baseline.json \
  --subset_ratio 0.1
