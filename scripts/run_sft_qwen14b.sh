#!/usr/bin/env bash
# Qwen-14B SFT on Game-24 (token-matched DAgger baseline): train + eval.
# DDP across all visible GPUs (or auto-detect free ones).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2,4 bash scripts/run_sft_qwen14b.sh [SEED]
#
# Env overrides: MEM_THRESHOLD (MiB), MASTER_PORT, CONFIG.
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${1:-1234}"
MEM_THRESHOLD="${MEM_THRESHOLD:-30000}"
MASTER_PORT="${MASTER_PORT:-29580}"
CONFIG="${CONFIG:-configs/sft_24_qwen14b.yaml}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t {print $1}' | paste -sd,)
  if [ -z "$FREE" ]; then
    echo "No GPU has >${MEM_THRESHOLD} MiB free." >&2; exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$FREE"
fi
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "SFT-Qwen14B | GPUs=$CUDA_VISIBLE_DEVICES (n=$NUM_GPUS) | seed=$SEED | port=$MASTER_PORT"

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m src.train_sft_24_qwen --config "$CONFIG" --seed "$SEED"

OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['output_dir'])")
LORA="$OUT_DIR/lora"
TEST=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data']['test_data'])")
RES=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['eval']['output_dir'])")
mkdir -p "$RES"

echo "=== generating on all $NUM_GPUS GPUs (data-parallel) ==="
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
PIDS=()
for i in "${!GPULIST[@]}"; do
  G="${GPULIST[$i]}"
  CUDA_VISIBLE_DEVICES=$G python -m src.generate_sft_qwen \
    --base_model "$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model']['base_model'])")" \
    --lora_adapter "$LORA" \
    --test_data "$TEST" \
    --output "$RES/gen_shard${i}.jsonl" \
    --shard_rank "$i" --shard_world "$NUM_GPUS" &
  PIDS+=($!)
done
wait "${PIDS[@]}"

# Merge shards
cat "$RES"/gen_shard*.jsonl > "$RES/generations.jsonl"
rm -f "$RES"/gen_shard*.jsonl

python -m src.evaluate_24 \
  --input "$RES/generations.jsonl" \
  --output "$RES/metrics.json"

echo "=== sft_qwen14b complete ==="
cat "$RES/metrics.json" | python -c "import sys,json; d=json.load(sys.stdin); print('accuracy:', d['overall'])"
