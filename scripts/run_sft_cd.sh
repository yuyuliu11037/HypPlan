#!/bin/bash
set -e

CONFIG="${1:-configs/sft_cd.yaml}"
echo "=== Countdown: CoT-SFT Training ==="
echo "Config: $CONFIG"

# Auto-detect free GPUs (>30 GB) unless CUDA_VISIBLE_DEVICES is already set
MEM_THRESHOLD=30000
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
           awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t {print $1}' | paste -sd,)
    export CUDA_VISIBLE_DEVICES="$FREE"
fi
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

# Step 1: Train
echo "--- Training ---"
if [ "$NUM_GPUS" -gt 1 ]; then
    python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=29501 \
        -m src.train_sft_cd --config "$CONFIG"
else
    python -m src.train_sft_cd --config "$CONFIG"
fi

# Step 2: Generate (single-GPU after merge)
echo "--- Generating ---"
ADAPTER=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['output_dir'])")
RESULTS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['eval']['output_dir'])")
TEST_DATA=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['test_data'])")
MAX_NEW=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['eval']['max_new_tokens'])")
TEMP=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['eval']['temperature'])")

FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print $1}')
CUDA_VISIBLE_DEVICES="$FIRST_GPU" python -m src.generate_cd_sft \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter "$ADAPTER" \
    --test_data "$TEST_DATA" \
    --output "$RESULTS_DIR/generations.jsonl" \
    --max_new_tokens "$MAX_NEW" \
    --temperature "$TEMP"

# Step 3: Evaluate
echo "--- Evaluating ---"
python -m src.evaluate_cd \
    --input "$RESULTS_DIR/generations.jsonl" \
    --output "$RESULTS_DIR/metrics.json"

echo "=== Done ==="
