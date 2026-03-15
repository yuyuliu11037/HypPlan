#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=4 src/evaluation/planning_eval.py \
  --model_name Qwen/Qwen2.5-7B \
  --proj_checkpoint checkpoints/stage1/proj_best.pt \
  --proj_type mlp \
  --structural_loss simple \
  --max_steps 20 \
  --max_step_tokens 256 \
  --output_file results/planning.json
