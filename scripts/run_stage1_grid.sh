#!/usr/bin/env bash
# Launch the 4-config stage-1 ablation grid in parallel.
# Configs: {poincare, lorentz} x {distortion, ranking}
# Each run uses ONE GPU (head training has tiny GPU footprint).
set -euo pipefail

cd "$(dirname "$0")/.."

MEM_THRESHOLD="${MEM_THRESHOLD:-10000}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -F',' -v t=$MEM_THRESHOLD '$2+0 > t {print $1}' | paste -sd,)
  if [ -z "$FREE" ]; then
    echo "No GPU has >${MEM_THRESHOLD} MiB free. Aborting." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$FREE"
fi
IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

CONFIGS=(
  "poincare distortion"
  "poincare ranking"
  "lorentz distortion"
  "lorentz ranking"
)

if [ $NUM_GPUS -lt ${#CONFIGS[@]} ]; then
  echo "Need ${#CONFIGS[@]} GPUs to run all 4 configs in parallel; have $NUM_GPUS" >&2
  echo "Will run sequentially on first GPU." >&2
fi

mkdir -p logs/stage1_grid
PIDS=()
for i in "${!CONFIGS[@]}"; do
  IFS=' ' read -r MANIFOLD LOSS <<< "${CONFIGS[$i]}"
  RUN_TAG="${MANIFOLD}_${LOSS}"
  GPU="${GPU_LIST[$((i % NUM_GPUS))]}"
  LOG="logs/stage1_grid/${RUN_TAG}.log"

  # Materialize config
  RUN_CONFIG="configs/head_${RUN_TAG}.yaml"
  python - <<PY
import yaml, pathlib
cfg = yaml.safe_load(open("configs/head.yaml"))
cfg["model"]["manifold"] = "$MANIFOLD"
cfg["training"]["loss"] = "$LOSS"
cfg["training"]["output_dir"] = f"checkpoints/head_$RUN_TAG"
cfg["eval"]["output_dir"] = f"results/head_eval/$RUN_TAG"
pathlib.Path("$RUN_CONFIG").write_text(yaml.dump(cfg))
PY

  echo "  launching $RUN_TAG on GPU $GPU -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -m src.train_head \
    --config "$RUN_CONFIG" > "$LOG" 2>&1 &
  PIDS+=($!)

  if [ $NUM_GPUS -lt ${#CONFIGS[@]} ]; then
    # Sequential fallback: wait for this one before next
    wait ${PIDS[-1]} || { echo "  $RUN_TAG FAILED"; exit 1; }
    echo "  $RUN_TAG done (seq)"
  fi
done

if [ $NUM_GPUS -ge ${#CONFIGS[@]} ]; then
  FAILED=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      IFS=' ' read -r M L <<< "${CONFIGS[$i]}"
      echo "  ${M}_${L}: ok"
    else
      IFS=' ' read -r M L <<< "${CONFIGS[$i]}"
      echo "  ${M}_${L}: FAILED"
      FAILED=1
    fi
  done
  [ $FAILED -eq 1 ] && exit 1
fi

# Evaluation pass (sequential; each uses 1 GPU briefly)
echo "=== evaluating all 4 heads ==="
for i in "${!CONFIGS[@]}"; do
  IFS=' ' read -r MANIFOLD LOSS <<< "${CONFIGS[$i]}"
  RUN_TAG="${MANIFOLD}_${LOSS}"
  RUN_CONFIG="configs/head_${RUN_TAG}.yaml"
  GPU="${GPU_LIST[$((i % NUM_GPUS))]}"
  EVAL_LOG="logs/stage1_grid/${RUN_TAG}_eval.log"
  echo "  eval $RUN_TAG on GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python -m src.eval_head \
    --config "$RUN_CONFIG" > "$EVAL_LOG" 2>&1
done

echo
echo "=== summary ==="
for i in "${!CONFIGS[@]}"; do
  IFS=' ' read -r MANIFOLD LOSS <<< "${CONFIGS[$i]}"
  RUN_TAG="${MANIFOLD}_${LOSS}"
  METRICS="results/head_eval/${RUN_TAG}/metrics.json"
  if [ -f "$METRICS" ]; then
    echo "--- $RUN_TAG ---"
    python -c "
import json
m = json.load(open('$METRICS'))
for split in ['val', 'test']:
    if split in m:
        s = m[split]
        print(f'  {split}: abs={s[\"mean_abs_distortion\"]:.3f} rel={s[\"mean_rel_distortion\"]:.3f} spearman={s[\"spearman\"]:.3f}')
"
  fi
done
