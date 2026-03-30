#!/bin/bash
# Stage 3: Joint LoRA fine-tuning with two-pass planning
# Uses torchrun for multi-GPU distributed training

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-configs/default.yaml}

echo "=== Stage 3: LoRA Fine-Tuning ==="
echo "GPUs: $NUM_GPUS | Config: $CONFIG"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    -m src.training.stage3 \
    --config "$CONFIG"
