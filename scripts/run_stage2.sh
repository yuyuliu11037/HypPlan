#!/bin/bash
# Stage 2: Structurize Proj with tree loss
# Uses torchrun for multi-GPU distributed training

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}
CONFIG=${CONFIG:-configs/default.yaml}

# Avoid NCCL issues with shared machines
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
MASTER_PORT=${MASTER_PORT:-29502}

echo "=== Stage 2: Structurize Proj ==="
echo "GPUs: $NUM_GPUS | Config: $CONFIG"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.training.stage2 \
    --config "$CONFIG"
