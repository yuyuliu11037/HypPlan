#!/usr/bin/env bash
# Train + eval the PT-SFT (Planning-Token SFT) baseline for one task,
# then commit + push the result.
#
# Usage:
#   bash scripts/run_pt_sft_pipeline.sh <task> [test_data]
#
# Tasks expected: numpath, rulechain, clutrr, proofwriter
# (G24/PQ/BW/GC have existing v1 checkpoints — run only eval there.)
#
# DDP: uses 4 GPUs from {1,2,3,4} for training (gloo), then 6-way
# sharded eval across {1,2,3,4,6,7}.
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

TASK=$1
DATA=${2:-data/${TASK}_test.jsonl}

CONFIG="configs/sft_pt_${TASK}_qwen14b.yaml"
LORA_DIR="checkpoints/sft_pt_${TASK}_qwen14b"
RESULT="results/baselines/${TASK}_ptsft.jsonl"
LOG_DIR="logs/baselines"
mkdir -p "$LOG_DIR" results/baselines

[ -f "$CONFIG" ] || { echo "MISSING $CONFIG"; exit 1; }
[ -f "$DATA" ] || { echo "MISSING $DATA"; exit 1; }

TORCHRUN=/data/yuyu/.local/bin/torchrun
export HYPPLAN_DIST_BACKEND=gloo

# ---- Train (skip if LoRA already exists) ----
if [ -d "$LORA_DIR/lora" ]; then
  echo "skip $TASK PT-SFT training (LoRA exists at $LORA_DIR/lora)"
else
  echo "=== ${TASK} PT-SFT training (DDP 4-GPU) ==="
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
    "$TORCHRUN" --nproc_per_node=4 --master_port=29503 \
      src/train_sft_pt_qwen.py --config "$CONFIG" \
      2>&1 | tee "$LOG_DIR/${TASK}_ptsft_train.log"
fi

# ---- Eval (sharded) ----
echo "=== ${TASK} PT-SFT eval (sharded 6-way) ==="
GPUS=(1 2 3 4 6 7)
SW=6
declare -a PIDS
for i in 0 1 2 3 4 5; do
  GPU=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$GPU nohup python3.10 -m src.eval_pt_ood \
    --task "$TASK" \
    --lora_adapter "$LORA_DIR/lora" \
    --test_data "$DATA" \
    --output "${RESULT%.jsonl}_shard${i}.jsonl" \
    --shard_rank $i --shard_world $SW \
    > "$LOG_DIR/${TASK}_ptsft_eval_shard${i}.log" 2>&1 &
  PIDS[$i]=$!
done
for pid in "${PIDS[@]}"; do wait $pid; done

cat "${RESULT%.jsonl}"_shard*.jsonl > "$RESULT"

# ---- Score ----
SUMMARY="${RESULT%.jsonl}.summary.txt"
python3.10 -m src.score_ood --task "$TASK" --input "$RESULT" \
  --show_failures 2 \
  | tee "$SUMMARY"

# ---- Commit + push ----
git add -f "$RESULT" "$SUMMARY"
git commit -m "$(cat <<EOF
baseline: ${TASK} PT-SFT (Planning-Token SFT) eval

Task: ${TASK}
$(grep -E "correct:" "$SUMMARY" | head -1)

Trained via src/train_sft_pt_qwen.py on ${TASK}_train_sft_plan.jsonl
(planning-tokens annotated). Eval via src/eval_pt_ood.py with
"Question: ...\\nAnswer:" prompt format matching the training.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" 2>&1 | tail -2
git push origin main 2>&1 | tail -2
echo "=== ${TASK} PT-SFT done ==="
