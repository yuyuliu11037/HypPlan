#!/bin/bash
# Evaluate Stage 1 model
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
NUM_GPUS=${NUM_GPUS:-6}
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
