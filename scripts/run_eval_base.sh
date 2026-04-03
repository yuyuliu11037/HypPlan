#!/bin/bash
# Evaluate plain Qwen2.5-7B base model (no training, no planning)
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
NUM_GPUS=${NUM_GPUS:-6}
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
