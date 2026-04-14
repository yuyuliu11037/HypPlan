#!/bin/bash
# Evaluate Stage 1 model
set -e

# Prompt for GPU selection
read -rp "Enter GPU IDs to use (comma-separated, e.g. 0,1,2): " GPU_INPUT
export CUDA_VISIBLE_DEVICES="$GPU_INPUT"
NUM_GPUS=$(echo "$GPU_INPUT" | awk -F',' '{print NF}')
CONFIG=${CONFIG:-configs/default.yaml}
CHECKPOINT=${CHECKPOINT:-checkpoints/stage1}
OUTPUT_DIR=results/eval

mkdir -p $OUTPUT_DIR

echo "=== Stage 1 Inference ==="
python -m src.generate_stage1 \
    --config $CONFIG \
    --checkpoint_dir $CHECKPOINT \
    --output $OUTPUT_DIR/stage1_generations.jsonl \
    --num_gpus $NUM_GPUS

echo "=== Stage 1 Evaluation ==="
python -m src.evaluate \
    --input $OUTPUT_DIR/stage1_generations.jsonl \
    --output $OUTPUT_DIR/stage1_metrics.json
