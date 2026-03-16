#!/usr/bin/env bash
set -euo pipefail

deepspeed --num_gpus=4 --module src.train_stage1 \
  --data_path data/prm800k_splits/train.jsonl \
  --presplit_data \
  --model_name Qwen/Qwen2.5-7B \
  --proj_type mlp \
  --structural_loss simple \
  --lambda_seg 0.1 \
  --lambda_depth 0.1 \
  --reconstruct_mode contextual \
  --max_seq_len 2048 \
  --max_step_len 256 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_epochs 3 \
  --lr 2e-4 \
  --output_dir checkpoints/stage1 \
  --deepspeed configs/deepspeed_config.json

