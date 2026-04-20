#!/usr/bin/env bash
# Data-parallel few-shot baseline: one model copy per detected free GPU,
# problems split round-robin across ranks.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
OUT_DIR="${OUT_DIR:-results/fewshot_qwen14b}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:--1}"
MEM_THRESHOLD="${MEM_THRESHOLD:-30000}"

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
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

mkdir -p "$OUT_DIR" logs/fewshot
PIDS=()
for i in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$i]}"
  LOG="logs/fewshot/shard_${i}.log"
  OUT="$OUT_DIR/generations_shard${i}.jsonl"
  echo "  shard $i on GPU $GPU → $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/fewshot_baseline.py \
    --model "$MODEL" \
    --output "$OUT" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --shard_rank "$i" \
    --shard_world "$NUM_GPUS" \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  shard $i: ok"
  else
    echo "  shard $i: FAILED (see logs/fewshot/shard_${i}.log)"
    FAILED=1
  fi
done

# Aggregate shard outputs
echo ""
echo "=== Aggregated ==="
python -c "
import glob, json
files = sorted(glob.glob('$OUT_DIR/generations_shard*.jsonl'))
records = []
for f in files:
    with open(f) as g:
        for line in g:
            records.append(json.loads(line))
n = len(records)
n_correct = sum(r['valid'] for r in records)
n_format = sum(r['format_ok'] for r in records)
print(f'Total: {n} problems')
print(f'Accuracy:     {n_correct/n:.4f} ({n_correct}/{n})')
print(f'Format-valid: {n_format/n:.4f} ({n_format}/{n})')

# Combined file
with open('$OUT_DIR/generations.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + chr(10))
print(f'Combined: $OUT_DIR/generations.jsonl')

with open('$OUT_DIR/metrics.json', 'w') as f:
    json.dump({
        'model': '$MODEL', 'n': n,
        'accuracy': n_correct/n, 'format_valid_rate': n_format/n,
        'n_correct': n_correct, 'n_format_ok': n_format,
    }, f, indent=2)
print(f'Metrics:  $OUT_DIR/metrics.json')
"

exit $FAILED
