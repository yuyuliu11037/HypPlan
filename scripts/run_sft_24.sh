#!/bin/bash
set -e

CONFIG="${1:-configs/sft_24.yaml}"
echo "=== Game of 24: CoT-SFT Training ==="
echo "Config: $CONFIG"

# Step 1: Train
echo "--- Training ---"
python -m src.train_sft_24 --config "$CONFIG"

# Step 2: Generate
echo "--- Generating ---"
python -m src.generate_24_sft \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter "checkpoints/sft_24_tot" \
    --test_data "data/24_test_tot.jsonl" \
    --output "results/24_sft_tot/generations.jsonl" \
    --max_new_tokens 256 \
    --temperature 0.0

# Step 3: Evaluate
echo "--- Evaluating ---"
python -m src.evaluate_24 \
    --input "results/24_sft_tot/generations.jsonl" \
    --output "results/24_sft_tot/metrics.json"

echo "=== Done ==="
