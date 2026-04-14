#!/bin/bash
# Evaluate plain Qwen2.5-7B base model (no training, no planning)
set -e

# Prompt for GPU selection
read -rp "Enter GPU IDs to use (comma-separated, e.g. 0,1,2): " GPU_INPUT
export CUDA_VISIBLE_DEVICES="$GPU_INPUT"
NUM_GPUS=$(echo "$GPU_INPUT" | awk -F',' '{print NF}')
CONFIG=${CONFIG:-configs/default.yaml}
OUTPUT_DIR=results/eval

mkdir -p $OUTPUT_DIR

echo "=== Base Model Inference ==="
python -m src.generate_base \
    --config $CONFIG \
    --output $OUTPUT_DIR/base_generations.jsonl \
    --num_gpus $NUM_GPUS

echo "=== Base Model Evaluation ==="
python -m src.evaluate \
    --input $OUTPUT_DIR/base_generations.jsonl \
    --output $OUTPUT_DIR/base_metrics.json
