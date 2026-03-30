#!/bin/bash
# Run inference + evaluation pipeline

set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}
STAGE3_DIR=${STAGE3_DIR:-checkpoints/stage3}
INPUT=${INPUT:-results/math_filtered.jsonl}
GEN_OUTPUT=${GEN_OUTPUT:-results/eval/hypplan_generations.jsonl}
EVAL_OUTPUT=${EVAL_OUTPUT:-results/eval/hypplan_metrics.json}
MAX_TOKENS=${MAX_TOKENS:-2048}

echo "=== HypPlan Inference ==="
python -m src.inference.generate \
    --config "$CONFIG" \
    --stage3_dir "$STAGE3_DIR" \
    --input "$INPUT" \
    --output "$GEN_OUTPUT" \
    --max_new_tokens "$MAX_TOKENS" \
    --temperature 0.0

echo "=== HypPlan Evaluation ==="
python -m src.eval.evaluate \
    --input "$GEN_OUTPUT" \
    --output "$EVAL_OUTPUT"
