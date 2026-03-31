#!/bin/bash
# CoT-SFT baseline: standard LoRA fine-tuning without planning tokens
# Uses torchrun for multi-GPU distributed training

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-5}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}
CONFIG=${CONFIG:-configs/default.yaml}

# Avoid NCCL issues with shared machines
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
MASTER_PORT=${MASTER_PORT:-29503}

echo "=== CoT-SFT Baseline ==="
echo "GPUs: $NUM_GPUS | Config: $CONFIG"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.training.cot_sft \
    --config "$CONFIG"
