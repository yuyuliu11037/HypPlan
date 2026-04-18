#!/usr/bin/env bash
# Stage-3 (DAgger) trainer for a SINGLE arm + inference + eval.
#
# Usage:
#   bash scripts/run_train_stage2_dagger.sh <ARM> [HEAD_TAG]
#     ARM:      "noz" or "z"
#     HEAD_TAG: e.g. "poincare_origin_ranking" (default).
#
# DDP across all GPUs in CUDA_VISIBLE_DEVICES (or auto-detected free ones).
# Intended to be launched twice in parallel with different GPU sets for the
# two arms, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_train_stage2_dagger.sh z   poincare_origin_ranking &
#   CUDA_VISIBLE_DEVICES=4,5,6   bash scripts/run_train_stage2_dagger.sh noz poincare_origin_ranking &
set -euo pipefail

cd "$(dirname "$0")/.."

ARM="${1:?usage: $0 <noz|z> [HEAD_TAG]}"
HEAD_TAG="${2:-poincare_origin_ranking}"
MEM_THRESHOLD="${MEM_THRESHOLD:-30000}"
MASTER_PORT="${MASTER_PORT:-29540}"
BASE_CONFIG="configs/stage2_dagger.yaml"
RUN_CONFIG="configs/stage2_dagger_${HEAD_TAG}.yaml"

if [ "$ARM" != "noz" ] && [ "$ARM" != "z" ]; then
  echo "ARM must be 'noz' or 'z', got '$ARM'" >&2
  exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t {print $1}' | paste -sd,)
  if [ -z "$FREE" ]; then
    echo "No GPU has >${MEM_THRESHOLD} MiB free. Aborting." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$FREE"
fi
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "ARM=$ARM | head_tag=$HEAD_TAG | GPUs=$CUDA_VISIBLE_DEVICES (n=$NUM_GPUS) | port=$MASTER_PORT"

# Materialize the per-run config (same config for both arms — they share it).
python - <<PY
import yaml, pathlib
cfg = yaml.safe_load(open("$BASE_CONFIG"))
cfg["model"]["head_checkpoint"] = f"checkpoints/head_${HEAD_TAG}/head.pt"
cfg["training"]["output_dir"] = f"checkpoints/dagger_stage2_${HEAD_TAG}"
cfg["eval"]["output_dir"] = f"results/dagger_stage2_${HEAD_TAG}"
pathlib.Path("$RUN_CONFIG").write_text(yaml.dump(cfg))
PY

USE_Z_FLAG=""
if [ "$ARM" = "z" ]; then USE_Z_FLAG="--use_z"; fi

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m src.train_stage2_dagger --config "$RUN_CONFIG" $USE_Z_FLAG --arm_tag "$ARM"

# Generate + evaluate (single GPU — the first visible one)
CKPT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['training']['output_dir'])")
RES_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['eval']['output_dir'])")
TEST=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['data']['test_data'])")
CKPT="$CKPT_ROOT/$ARM"
RES="$RES_ROOT/$ARM"
mkdir -p "$RES"
EXTRA=""
if [ "$ARM" = "noz" ]; then EXTRA="--no_z_inject"; fi

echo "=== generating arm=$ARM ==="
CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1}') \
  python -m src.generate_24_stage2 \
    --stage2_checkpoint "$CKPT" \
    --test_data "$TEST" \
    --output "$RES/generations.jsonl" $EXTRA

python -m src.evaluate_24 \
  --input "$RES/generations.jsonl" \
  --output "$RES/metrics.json"

echo "=== arm=$ARM complete ==="
cat "$RES/metrics.json" | python -c "import sys,json; d=json.load(sys.stdin); print('accuracy:', d['overall'])"
