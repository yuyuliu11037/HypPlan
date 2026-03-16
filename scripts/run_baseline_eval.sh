#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=4 -m src.evaluation.baseline_eval \
  --model_name Qwen/Qwen2.5-7B \
  --local_eval_path data/prm800k_splits/eval.jsonl \
  --max_new_tokens 1024 \
  --output_file results/baseline.json

