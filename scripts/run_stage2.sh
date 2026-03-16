#!/usr/bin/env bash
set -euo pipefail

deepspeed --num_gpus=4 --module src.train_stage2 \
  --data_path data/prm800k_splits/train.jsonl \
  --presplit_data \
  --model_name Qwen/Qwen2.5-7B \
  --proj_checkpoint checkpoints/stage1/proj.pt \
  --proj_type mlp \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj v_proj \
  --lora_dropout 0.05 \
  --max_seq_len 2048 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_epochs 2 \
  --lr 1e-4 \
  --output_dir checkpoints/stage2 \
  --deepspeed configs/deepspeed_config.json

