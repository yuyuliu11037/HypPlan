#!/bin/bash
set -e

CONFIG="${1:-configs/plan_24.yaml}"
echo "=== Game of 24: Planning Vector Training ==="

# Step 1: Train ProjMLP
echo "--- Training ProjMLP ---"
python -m src.train_plan_24 --config "$CONFIG"

# Step 2: Generate with planning vectors
echo "--- Generating (SFT + planning) ---"
python -m src.generate_24_plan \
    --base_model "checkpoints/sft_24_v2_merged" \
    --proj_checkpoint "checkpoints/plan_24" \
    --test_data "data/24_test.jsonl" \
    --output "results/24_plan/generations.jsonl" \
    --max_new_tokens 256 \
    --temperature 0.0

# Step 3: Evaluate
echo "--- Evaluating ---"
python -m src.evaluate_24 \
    --input "results/24_plan/generations.jsonl" \
    --output "results/24_plan/metrics.json"

echo "=== Done ==="
