#!/usr/bin/env bash
# Launcher for varied-target DAgger training.
# Auto-detects free GPUs and launches torchrun with manual DDP.
# NCCL-safe: avoids putting both GPU 5 and GPU 7 in the same process group
# (broken pair on this host, noted in README).
#
# Usage:
#   bash scripts/run_train_stage2_dagger_varied.sh [z|noz|randz] [seed]
# Defaults: arm=z, seed=1234
set -euo pipefail

ARM=${1:-z}
SEED=${2:-1234}
CONFIG=${CONFIG:-configs/stage2_dagger_24_varied_qwen14b.yaml}

case "$ARM" in
  z)     USE_Z_FLAG="--use_z"; RANDOMZ_FLAG="" ;;
  noz)   USE_Z_FLAG="";        RANDOMZ_FLAG="" ;;
  randz) USE_Z_FLAG="--use_z"; RANDOMZ_FLAG="--random_z" ;;
  *) echo "arm must be z|noz|randz"; exit 1 ;;
esac

MEM_THRESHOLD=${MEM_THRESHOLD:-30000}  # MiB free to count as available
FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
       awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t+0 {print $1+0}')

# Drop GPU 7 if GPU 5 is already in the candidate list (NCCL-broken pair).
have5=$(echo "$FREE" | grep -c '^5$' || true)
if [ "$have5" = "1" ]; then
  FREE=$(echo "$FREE" | grep -v '^7$')
fi

# Default cap 6; override with MAX_GPUS env.
MAX_GPUS=${MAX_GPUS:-6}
SELECTED=$(echo "$FREE" | head -n $MAX_GPUS | paste -sd,)
if [ -z "$SELECTED" ]; then
  echo "No free GPUs with >= $MEM_THRESHOLD MiB"; exit 1
fi

export CUDA_VISIBLE_DEVICES="$SELECTED"
NUM_GPUS=$(echo "$SELECTED" | awk -F',' '{print NF}')
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES  (arm=$ARM seed=$SEED)"

LOG_DIR=results/logs/phase4_dagger_varied
mkdir -p "$LOG_DIR"

PORT=${PORT:-29640}
TORCHRUN=${TORCHRUN:-/data/yuyu/.local/bin/torchrun}
"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
         --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT \
  -m src.train_stage2_dagger_varied \
    --config "$CONFIG" \
    $USE_Z_FLAG $RANDOMZ_FLAG \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/${ARM}_s${SEED}.log"
