#!/usr/bin/env bash
# Generate Countdown tree data (sampled history-subtree + hidden states).
#
# Shards problems across all detected free GPUs (one process per GPU).
# Each process is independent, writes its own file set, and skips files
# already present so re-runs are resume-safe.
set -euo pipefail

cd "$(dirname "$0")/.."

MEM_THRESHOLD="${MEM_THRESHOLD:-20000}"
LIMIT="${LIMIT:--1}"
BASE_MODEL="${BASE_MODEL:-checkpoints/sft_cd_merged}"
OUT_DIR="${OUT_DIR:-data/cd_trees}"
LOG_DIR="${LOG_DIR:-logs/gen_tree_cd}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_TRAJECTORIES="${N_TRAJECTORIES:-200}"
N_GUIDED="${N_GUIDED:-20}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t {print $1}' | paste -sd,)
  if [ -z "$FREE" ]; then
    echo "No GPU has >${MEM_THRESHOLD} MiB free. Aborting." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$FREE"
fi
IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

mkdir -p "$LOG_DIR"
PIDS=()
for i in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$i]}"
  LOG="$LOG_DIR/shard_${i}.log"
  echo "  launching shard $i on GPU $GPU → $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python data/generate_tree_data_cd.py \
    --base_model "$BASE_MODEL" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --n_trajectories "$N_TRAJECTORIES" \
    --n_guided "$N_GUIDED" \
    --limit "$LIMIT" \
    --shard_rank "$i" \
    --shard_world "$NUM_GPUS" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  shard $i: ok"
  else
    echo "  shard $i: FAILED (see $LOG_DIR/shard_${i}.log)"
    FAILED=1
  fi
done
exit $FAILED
