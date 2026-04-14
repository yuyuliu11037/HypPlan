#!/bin/bash
set -e

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
NUM_SHOTS="${2:-0}"
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR="results/24_${MODEL_SHORT}_${NUM_SHOTS}shot"
mkdir -p "$OUTPUT_DIR"

echo "=== Game of 24: ${NUM_SHOTS}-shot baseline with $MODEL ==="

# Step 1: Generate
echo "Generating..."
python -m src.generate_24_zeroshot \
    --model "$MODEL" \
    --test_data data/24_test.jsonl \
    --train_data data/24_train.jsonl \
    --output "$OUTPUT_DIR/generations.jsonl" \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --num_shots "$NUM_SHOTS"

# Step 2: Evaluate
echo "Evaluating..."
python -m src.evaluate_24 \
    --input "$OUTPUT_DIR/generations.jsonl" \
    --output "$OUTPUT_DIR/metrics.json"

echo "Done. Results in $OUTPUT_DIR/"
