#!/usr/bin/env bash
# Stage-3 (DAgger) trainer for a SINGLE arm + inference + eval.
#
# Usage:
#   bash scripts/run_train_stage2_dagger.sh <ARM> [HEAD_TAG] [SEED]
#     ARM:      "noz" or "z"
#     HEAD_TAG: e.g. "poincare_origin_ranking" (default).
#     SEED:     optional int. When set, output subdir becomes
#               "{arm}_s{seed}" so multi-seed runs don't overwrite each other.
#
# DDP across all GPUs in CUDA_VISIBLE_DEVICES (or auto-detected free ones).
# Intended to be launched in parallel for different (arm, seed) combos:
#   CUDA_VISIBLE_DEVICES=0,5  bash scripts/run_train_stage2_dagger.sh z   poincare_origin_ranking 1234 &
#   CUDA_VISIBLE_DEVICES=1,7  bash scripts/run_train_stage2_dagger.sh noz poincare_origin_ranking 1234 &
set -euo pipefail

cd "$(dirname "$0")/.."

ARM="${1:?usage: $0 <noz|z> [HEAD_TAG] [SEED]}"
HEAD_TAG="${2:-poincare_origin_ranking}"
SEED="${3:-}"
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
SEED_FLAG=""
ARM_SUFFIX="$ARM"
if [ -n "$SEED" ]; then
  SEED_FLAG="--seed $SEED"
  ARM_SUFFIX="${ARM}_s${SEED}"
fi

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m src.train_stage2_dagger --config "$RUN_CONFIG" $USE_Z_FLAG $SEED_FLAG --arm_tag "$ARM_SUFFIX"

# Generate + evaluate (single GPU — the first visible one)
CKPT_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['training']['output_dir'])")
RES_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['eval']['output_dir'])")
TEST=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['data']['test_data'])")
CKPT="$CKPT_ROOT/$ARM_SUFFIX"
RES="$RES_ROOT/$ARM_SUFFIX"
mkdir -p "$RES"
EXTRA=""
if [ "$ARM" = "noz" ]; then EXTRA="--no_z_inject"; fi

echo "=== generating arm=$ARM_SUFFIX ==="
CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1}') \
  python -m src.generate_24_stage2 \
    --stage2_checkpoint "$CKPT" \
    --test_data "$TEST" \
    --output "$RES/generations.jsonl" $EXTRA

python -m src.evaluate_24 \
  --input "$RES/generations.jsonl" \
  --output "$RES/metrics.json"

echo "=== arm=$ARM_SUFFIX complete ==="
cat "$RES/metrics.json" | python -c "import sys,json; d=json.load(sys.stdin); print('accuracy:', d['overall'])"
