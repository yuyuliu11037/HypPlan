#!/usr/bin/env bash
# Generate tree data for HypPlan v2.
#
# Shards problems across all detected free GPUs (one process per GPU).
# Each process is independent, writes its own file set, and skips files
# already present so re-runs are resume-safe.
set -euo pipefail

cd "$(dirname "$0")/.."

MEM_THRESHOLD="${MEM_THRESHOLD:-20000}"
LIMIT="${LIMIT:--1}"
BASE_MODEL="${BASE_MODEL:-checkpoints/sft_24_tot_merged}"
BATCH_SIZE="${BATCH_SIZE:-64}"

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

mkdir -p logs/gen_tree
PIDS=()
for i in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$i]}"
  LOG="logs/gen_tree/shard_${i}.log"
  echo "  launching shard $i on GPU $GPU → $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python data/generate_tree_data.py \
    --base_model "$BASE_MODEL" \
    --out_dir data/trees \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --shard_rank "$i" \
    --shard_world "$NUM_GPUS" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

# Wait for all shards and report per-shard status
FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  shard $i: ok"
  else
    echo "  shard $i: FAILED (see logs/gen_tree/shard_${i}.log)"
    FAILED=1
  fi
done
exit $FAILED
