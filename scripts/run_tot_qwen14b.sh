#!/usr/bin/env bash
# Launch ToT baseline on Qwen-2.5-14B-Instruct (both generator and evaluator).
# Single-model setup via --shared_model. Uses chat template.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
OUT_DIR="${OUT_DIR:-results/tot_baseline_qwen14b/seed_1234}"
SEED="${SEED:-1234}"
TEMPERATURE="${TEMPERATURE:-0.7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:--1}"
MEM_THRESHOLD="${MEM_THRESHOLD:-30000}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t {print $1}' | paste -sd,)
  if [ -z "$FREE" ]; then
    echo "No GPU has >${MEM_THRESHOLD} MiB free. Aborting." >&2
    exit 1
  fi
  # ToT only needs one GPU for 14B bf16 (~28 GB). Pick first free.
  export CUDA_VISIBLE_DEVICES="$(echo "$FREE" | cut -d',' -f1)"
fi
echo "Using GPU: $CUDA_VISIBLE_DEVICES | model=$MODEL | seed=$SEED"

mkdir -p "$OUT_DIR"
python -m src.tot_baseline \
  --generator "$MODEL" \
  --shared_model \
  --use_chat_template \
  --test_data data/24_test_tot.jsonl \
  --output_dir "$OUT_DIR" \
  --seed "$SEED" \
  --temperature "$TEMPERATURE" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT"
