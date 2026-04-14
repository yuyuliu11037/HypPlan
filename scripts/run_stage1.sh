#!/bin/bash
# Stage 1: Warm up Proj with frozen LLM
set -euo pipefail

# Prompt for GPU selection
read -rp "Enter GPU IDs to use (comma-separated, e.g. 0,1,2): " GPU_INPUT
export CUDA_VISIBLE_DEVICES="$GPU_INPUT"
NUM_GPUS=$(echo "$GPU_INPUT" | awk -F',' '{print NF}')
CONFIG=${CONFIG:-configs/default.yaml}

export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
MASTER_PORT=${MASTER_PORT:-29501}

echo "=== Stage 1: Warm Up Proj ==="
echo "GPUs: $NUM_GPUS | Config: $CONFIG"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.train_stage1 \
    --config "$CONFIG"
