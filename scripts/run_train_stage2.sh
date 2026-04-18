#!/usr/bin/env bash
# Stage-2 trainer + inference + evaluator.
#
# Usage:
#   bash scripts/run_train_stage2.sh [HEAD_TAG]
#     HEAD_TAG: e.g. "poincare_distortion" (default). Must match a prior
#               run of run_train_head.sh that produced
#               checkpoints/head_${HEAD_TAG}/head.pt.
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_TAG="${1:-poincare_distortion}"
MEM_THRESHOLD="${MEM_THRESHOLD:-15000}"
MASTER_PORT="${MASTER_PORT:-29511}"

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
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES | head_tag=$HEAD_TAG"

RUN_CONFIG="configs/stage2_${HEAD_TAG}.yaml"
python - <<PY
import yaml, pathlib
cfg = yaml.safe_load(open("configs/stage2.yaml"))
cfg["model"]["head_checkpoint"] = f"checkpoints/head_${HEAD_TAG}/head.pt"
manifold = "$HEAD_TAG".split("_")[0]
cfg["model"]["manifold"] = manifold
cfg["training"]["output_dir"] = f"checkpoints/hyp_stage2_${HEAD_TAG}"
cfg["eval"]["output_dir"] = f"results/hyp_stage2_${HEAD_TAG}"
pathlib.Path("$RUN_CONFIG").write_text(yaml.dump(cfg))
print("wrote $RUN_CONFIG")
PY

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
  -m src.train_stage2 --config "$RUN_CONFIG"

OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['eval']['output_dir'])")
CKPT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['training']['output_dir'])")
TEST=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['data']['test_data'])")
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print $1}') \
  python -m src.generate_24_stage2 \
    --stage2_checkpoint "$CKPT_DIR" \
    --test_data "$TEST" \
    --output "$OUT_DIR/generations.jsonl"

python -m src.evaluate_24 \
  --input "$OUT_DIR/generations.jsonl" \
  --output "$OUT_DIR/metrics.json"
