#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=4 -m src.evaluation.planning_eval \
  --model_name Qwen/Qwen2.5-7B \
  --local_eval_path data/prm800k_splits/eval.jsonl \
  --prompt_style train_compatible \
  --lora_checkpoint checkpoints/stage2/lora_adapters \
  --proj_checkpoint checkpoints/stage2/proj.pt \
  --proj_type mlp \
  --inference_mode autonomous \
  --max_new_tokens 1024 \
  --output_file results/stage2_autonomous.json

