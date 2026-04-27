#!/usr/bin/env bash
# Run one (task, mode) baseline cell, sharded across N GPUs, then
# concat → score → commit → push.
#
# Usage:
#   bash scripts/run_one_baseline.sh <task> <mode> [shard_world] [test_data] [extra_args...]
#
# Examples:
#   bash scripts/run_one_baseline.sh numpath tot 6
#   bash scripts/run_one_baseline.sh proofwriter sc 6
#   bash scripts/run_one_baseline.sh g24 tot 6 data/24_test.jsonl --limit 100
#
# Outputs:
#   logs/baselines/{task}_{mode}_shard{i}.log
#   results/baselines/{task}_{mode}.jsonl  (concatenated)
#   results/baselines/{task}_{mode}.summary.txt
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

TASK=$1; shift
MODE=$1; shift
SW=${1:-6}; shift || true
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
  DATA=$1; shift
else
  DATA="data/${TASK}_test.jsonl"
fi
EXTRA="$@"

OUT_PATH="results/baselines/${TASK}_${MODE}.jsonl"
LOG_DIR="logs/baselines"
mkdir -p "$LOG_DIR" results/baselines

GPUS=(1 2 3 4 6 7)
echo "=== ${TASK} ${MODE} (sharded ${SW}-way) ==="

# Launch shards
declare -a PIDS
for i in $(seq 0 $((SW-1))); do
  GPU=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$GPU nohup python3.10 -m src.eval_baseline_kpath \
    --task "$TASK" --mode "$MODE" \
    --test_data "$DATA" \
    --out_path "$OUT_PATH" \
    --K 5 --temperature 0.7 --max_new_tokens 384 \
    --shard_rank $i --shard_world "$SW" \
    $EXTRA \
    > "$LOG_DIR/${TASK}_${MODE}_shard${i}.log" 2>&1 &
  PIDS[$i]=$!
done

# Wait for all
for pid in "${PIDS[@]}"; do
  wait $pid
done

# Concat shards
SHARD_FILES=( "${OUT_PATH%.jsonl}"_shard*.jsonl )
cat "${SHARD_FILES[@]}" > "$OUT_PATH"

# Score (extract aggregate numbers)
SUMMARY_PATH="${OUT_PATH%.jsonl}.summary.txt"
python3.10 - <<PY > "$SUMMARY_PATH"
import json
records = [json.loads(l) for l in open("$OUT_PATH")]
n = len(records)
top1 = sum(1 for r in records if r.get("top1_ok"))
maj = sum(1 for r in records if r.get("majority_ok"))
print(f"task=${TASK} mode=${MODE} n={n}")
if any(r.get("top1_gen") is not None for r in records):
    print(f"  top1: {top1}/{n} = {top1/n:.0%}")
if "$MODE" == "sc":
    print(f"  majority: {maj}/{n} = {maj/n:.0%}")
PY

cat "$SUMMARY_PATH"

# Commit + push
git add -f "$OUT_PATH" "$SUMMARY_PATH"
git commit -m "$(cat <<EOF
baseline: ${TASK} ${MODE} eval ($(grep -oP '\d+/\d+ = \d+%' "$SUMMARY_PATH" | head -3 | paste -sd '; '))

$(cat "$SUMMARY_PATH")

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" 2>&1 | tail -2

git push origin main 2>&1 | tail -2

echo "=== ${TASK} ${MODE} done ==="
