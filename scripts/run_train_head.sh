#!/usr/bin/env bash
# Stage-1 head trainer + evaluator.
#
# Usage:
#   bash scripts/run_train_head.sh [MANIFOLD] [LOSS]
#     MANIFOLD: "poincare" (default) | "lorentz"
#     LOSS:     "distortion" (default) | "ranking"
#
# Auto-detects free GPUs (>MEM_THRESHOLD MiB) and runs DDP across them.
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFOLD="${1:-poincare}"
LOSS="${2:-distortion}"
RUN_TAG="${MANIFOLD}_${LOSS}"
MEM_THRESHOLD="${MEM_THRESHOLD:-15000}"
MASTER_PORT="${MASTER_PORT:-29510}"
BASE_CONFIG="${BASE_CONFIG:-configs/head.yaml}"

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
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES | manifold=$MANIFOLD loss=$LOSS"

# Materialize a per-run config by editing the three fields.
RUN_CONFIG="configs/head_${RUN_TAG}.yaml"
python - <<PY
import yaml, pathlib
cfg = yaml.safe_load(open("$BASE_CONFIG"))
cfg["model"]["manifold"] = "$MANIFOLD"
cfg["training"]["loss"] = "$LOSS"
cfg["training"]["output_dir"] = f"checkpoints/head_$RUN_TAG"
cfg["eval"]["output_dir"] = f"results/head_eval/$RUN_TAG"
pathlib.Path("$RUN_CONFIG").write_text(yaml.dump(cfg))
print("wrote $RUN_CONFIG")
PY

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m src.train_head --config "$RUN_CONFIG"

python -m src.eval_head --config "$RUN_CONFIG"
