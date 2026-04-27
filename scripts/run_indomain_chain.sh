#!/usr/bin/env bash
# Train + eval HypPlan in-domain (Stage-1 head + Stage-2 LoRA + eval)
# for the 4 missing tasks: clutrr, rulechain, numpath, proofwriter.
#
# Strategy:
#   - tree-data for numpath + proofwriter: parallel background on GPUs 1, 2
#     (each ~1-2 hr; uses Qwen-14B forward).
#   - foreground sequential pipeline on GPUs 3,4,6,7:
#       1. head_rulechain (DDP gloo, ~15 min)
#       2. clutrr Stage-2 LoRA (DDP gloo, ~30 min) + eval (~5 min)
#       3. rulechain Stage-2 LoRA (DDP gloo, ~30 min) + eval (~5 min)
#   - then wait for tree-data and continue:
#       4. head_numpath
#       5. numpath Stage-2 + eval
#       6. head_proofwriter
#       7. proofwriter Stage-2 + eval
#
# Each phase commits + pushes its results.
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

LOG_DIR="logs/indomain"
mkdir -p "$LOG_DIR" results/eval_stage2_indomain

TORCHRUN=/data/yuyu/.local/bin/torchrun
export HYPPLAN_DIST_BACKEND=gloo

# ---------- Helper: run sharded eval + commit ----------
eval_indomain() {
  local task=$1
  local lora_dir=$2
  local head_path=$3
  local test_data=$4
  local out_dir="results/eval_stage2_indomain/${task}"
  mkdir -p "$out_dir"
  local result="${out_dir}/${task}.jsonl"
  if [ -s "$result" ]; then
    echo "skip $task in-domain eval (exists)"
    return
  fi
  echo "=== ${task} in-domain eval ==="
  GPUS=(1 2 3 4 6 7)
  declare -a PIDS
  for i in 0 1 2 3 4 5; do
    local GPU=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU nohup python3.10 -m src.eval_stage2_indomain \
      --task "$task" \
      --ckpt_dir "$lora_dir" --head_path "$head_path" \
      --test_data "$test_data" \
      --output "${result%.jsonl}_shard${i}.jsonl" \
      --shard_rank $i --shard_world 6 \
      > "$LOG_DIR/${task}_indomain_eval_shard${i}.log" 2>&1 &
    PIDS[$i]=$!
  done
  for pid in "${PIDS[@]}"; do wait $pid; done
  cat "${result%.jsonl}"_shard*.jsonl > "$result"

  python3.10 - <<PY > "${out_dir}/summary.txt"
import json
recs = [json.loads(l) for l in open("$result")]
n = len(recs)
correct = sum(1 for r in recs if r.get("correct"))
print(f"task=$task in-domain HypPlan n={n}")
print(f"  correct: {correct}/{n} = {correct/n:.0%}")
PY
  cat "${out_dir}/summary.txt"

  git add -f "$result" "${out_dir}/summary.txt"
  git commit -m "$(printf 'indomain HypPlan: %s eval (%s)\n\n%s\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' \
      "$task" "$(grep -oE '\d+/\d+ = \d+%' "${out_dir}/summary.txt" | head -1)" "$(cat "${out_dir}/summary.txt")")" \
    2>&1 | tail -2
  git push origin main 2>&1 | tail -2
}

# ---------- Helper: train Stage-1 head ----------
train_head() {
  local task=$1
  local config="configs/head_${task}_qwen14b_rank.yaml"
  local out="checkpoints/head_${task}_qwen14b_rank/head.pt"
  if [ -f "$out" ]; then
    echo "skip head_${task} (exists)"
    return
  fi
  [ -f "$config" ] || { echo "MISSING $config"; return 1; }
  echo "=== head_${task} training ==="
  CUDA_VISIBLE_DEVICES=3,4,6,7 \
    "$TORCHRUN" --nproc_per_node=4 --master_port=29504 \
      src/train_head.py --config "$config" \
      2>&1 | tee "$LOG_DIR/head_${task}.log"
}

# ---------- Helper: train Stage-2 in-domain LoRA ----------
train_stage2() {
  local task=$1
  local config=$2
  local out_dir=$3
  if [ -d "$out_dir/lora" ]; then
    echo "skip Stage-2 ${task} (exists)"
    return
  fi
  echo "=== Stage-2 ${task} training (DDP 4-GPU) ==="
  CUDA_VISIBLE_DEVICES=3,4,6,7 \
    "$TORCHRUN" --nproc_per_node=4 --master_port=29505 \
      src/train_stage2_dagger_ood.py \
        --task "$task" --config "$config" --use_z 1 \
      2>&1 | tee "$LOG_DIR/stage2_${task}.log"
}

# ---------- Phase 0: kick off long tree-data jobs in background ----------
NUMPATH_TREE_PID=""
PROOFWRITER_TREE_PID=""
if [ ! -d "data/numpath_trees_qwen14b/test" ] || \
   [ "$(ls data/numpath_trees_qwen14b/test/problem_*.pt 2>/dev/null | wc -l)" -lt 200 ]; then
  echo "=== launching numpath tree-data on GPU 1 (background) ==="
  CUDA_VISIBLE_DEVICES=1 nohup python -m data.generate_tree_data_groupB \
    --task numpath --splits train,val,test \
    > "$LOG_DIR/treedata_numpath.log" 2>&1 &
  NUMPATH_TREE_PID=$!
  echo "  PID $NUMPATH_TREE_PID"
fi
if [ ! -d "data/proofwriter_trees_qwen14b/test" ] || \
   [ "$(ls data/proofwriter_trees_qwen14b/test/problem_*.pt 2>/dev/null | wc -l)" -lt 200 ]; then
  echo "=== launching proofwriter tree-data on GPU 2 (background) ==="
  CUDA_VISIBLE_DEVICES=2 nohup python -m data.generate_tree_data_groupB \
    --task proofwriter --splits train,val,test \
    > "$LOG_DIR/treedata_proofwriter.log" 2>&1 &
  PROOFWRITER_TREE_PID=$!
  echo "  PID $PROOFWRITER_TREE_PID"
fi

# ---------- Phase 1: rulechain head + Stage-2 + eval ----------
train_head rulechain
train_stage2 rulechain configs/stage2_dagger_rulechain_indomain.yaml \
                       checkpoints/dagger_stage2_rulechain_indomain
eval_indomain rulechain checkpoints/dagger_stage2_rulechain_indomain \
              checkpoints/head_rulechain_qwen14b_rank/head.pt \
              data/rulechain_test.jsonl

# ---------- Phase 2: clutrr Stage-2 + eval (head already done) ----------
train_stage2 clutrr configs/stage2_dagger_clutrr_qwen14b.yaml \
                    checkpoints/dagger_stage2_clutrr_indomain
eval_indomain clutrr checkpoints/dagger_stage2_clutrr_indomain \
              checkpoints/head_clutrr_qwen14b_rank/head.pt \
              data/clutrr_test.jsonl

# ---------- Wait for tree-data background jobs ----------
if [ -n "$NUMPATH_TREE_PID" ]; then
  echo "=== waiting on numpath tree-data ==="
  wait "$NUMPATH_TREE_PID" || true
fi
if [ -n "$PROOFWRITER_TREE_PID" ]; then
  echo "=== waiting on proofwriter tree-data ==="
  wait "$PROOFWRITER_TREE_PID" || true
fi

# ---------- Phase 3: numpath head + Stage-2 + eval ----------
train_head numpath
train_stage2 numpath configs/stage2_dagger_numpath_qwen14b.yaml \
                     checkpoints/dagger_stage2_numpath_indomain
eval_indomain numpath checkpoints/dagger_stage2_numpath_indomain \
              checkpoints/head_numpath_qwen14b_rank/head.pt \
              data/numpath_test.jsonl

# ---------- Phase 4: proofwriter head + Stage-2 + eval ----------
train_head proofwriter
train_stage2 proofwriter configs/stage2_dagger_proofwriter_qwen14b.yaml \
                         checkpoints/dagger_stage2_proofwriter_indomain
eval_indomain proofwriter checkpoints/dagger_stage2_proofwriter_indomain \
              checkpoints/head_proofwriter_qwen14b_rank/head.pt \
              data/proofwriter_test.jsonl

echo "=== All in-domain HypPlan tests done ==="
