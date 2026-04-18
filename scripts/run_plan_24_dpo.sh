#!/bin/bash
set -e

CONFIG="${1:-configs/plan_24_dpo.yaml}"
echo "=== Game of 24: DPO Planning Vector Training ==="

# Step 1: Generate preference pairs (if not present)
if [ ! -f data/24_train_dpo_tot.jsonl ]; then
  echo "--- Generating DPO preference pairs ---"
  python data/generate_24_dpo_pairs.py
fi

# Step 2: Precompute reference log-probs (if not present)
if [ ! -f data/24_train_dpo_tot_refs.pt ]; then
  echo "--- Precomputing reference log-probs ---"
  CUDA_VISIBLE_DEVICES=0 python -m src.precompute_dpo_refs
fi

# Step 3: Train ProjMLP with DPO on 4 GPUs
echo "--- Training ProjMLP (DPO) ---"
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /data/yuyu/.local/bin/torchrun \
    --nproc_per_node=4 --master_port=29506 \
    -m src.train_plan_24_dpo --config "$CONFIG"

# Step 4: Generate with planning vectors (sharded)
echo "--- Generating (SFT + DPO planning) ---"
rm -rf results/24_plan_dpo
mkdir -p results/24_plan_dpo
for i in 0 1 2 3 4; do
  GPU=$((i))
  CUDA_VISIBLE_DEVICES=$GPU python -m src.generate_24_plan \
      --base_model checkpoints/sft_24_tot_merged \
      --proj_checkpoint checkpoints/plan_24_dpo \
      --test_data data/24_test_tot_shard${i}.jsonl \
      --output results/24_plan_dpo/generations_shard${i}.jsonl \
      --max_new_tokens 256 --temperature 0.0 &
done
wait
cat results/24_plan_dpo/generations_shard*.jsonl > results/24_plan_dpo/generations.jsonl

# Step 5: Evaluate
echo "--- Evaluating ---"
python -m src.evaluate_24 \
    --input results/24_plan_dpo/generations.jsonl \
    --output results/24_plan_dpo/metrics.json

echo "=== Done ==="
