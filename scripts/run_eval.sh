#!/bin/bash
# Run inference + evaluation pipeline

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}
NUM_GPUS=${NUM_GPUS:-5}

CONFIG=${CONFIG:-configs/default.yaml}
STAGE3_DIR=${STAGE3_DIR:-checkpoints/stage3}
INPUT=${INPUT:-results/math_filtered.jsonl}
GEN_OUTPUT=${GEN_OUTPUT:-results/eval/hypplan_generations.jsonl}
EVAL_OUTPUT=${EVAL_OUTPUT:-results/eval/hypplan_metrics.json}
MAX_TOKENS=${MAX_TOKENS:-2048}

echo "=== HypPlan Inference ($NUM_GPUS GPUs) ==="
python -m src.inference.generate \
    --config "$CONFIG" \
    --stage3_dir "$STAGE3_DIR" \
    --input "$INPUT" \
    --output "$GEN_OUTPUT" \
    --max_new_tokens "$MAX_TOKENS" \
    --temperature 0.0 \
    --num_gpus "$NUM_GPUS"

echo "=== HypPlan Evaluation ==="
python -m src.eval.evaluate \
    --input "$GEN_OUTPUT" \
    --output "$EVAL_OUTPUT"
