#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the base model with plain LoRA SFT on the first 8000 samples of
# prm800k_raw.jsonl, matching stage 2's LoRA config for a fair comparison.
deepspeed --num_gpus=4 --module src.train_baseline \
  --data_path data/prm800k_raw.jsonl \
  --limit 8000 \
  --model_name Qwen/Qwen2.5-7B \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj v_proj \
  --lora_dropout 0.05 \
  --max_seq_len 2048 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_epochs 2 \
  --lr 1e-4 \
  --output_dir checkpoints/baseline \
  --deepspeed configs/deepspeed_config.json

# Evaluate the fine-tuned baseline model.
torchrun --nproc_per_node=4 -m src.evaluation.baseline_eval \
  --model_name Qwen/Qwen2.5-7B \
  --lora_adapter_path checkpoints/baseline/lora_adapters \
  --local_eval_path data/prm800k_splits/eval.jsonl \
  --max_new_tokens 1024 \
  --output_file results/baseline.json
