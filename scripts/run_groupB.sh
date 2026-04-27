#!/usr/bin/env bash
# Master driver for the Group B replication pipeline.
#
# Phases (each one logs its own PIDs and waits before continuing):
#   0. Pre-flight: verify JSONL data + configs exist
#   1. Stage-1 heads (4 tasks, parallel across GPUs 1, 6, 7, 4)
#   2. Rule-chaining Stage-2 LoRA (the Group B task-agnostic LoRA, DDP 4-GPU)
#   3. OOD 4-cell eval matrix (3 tasks × 4 conditions, sharded)
#   4. Matched-prompt control on synthlogic
#   5. In-domain Stage-2 LoRAs (3 tasks, sequential to avoid GPU contention)
#   6. In-domain HypPlan eval (3 tasks)
#   7. PT-SFT baselines (4 tasks, sequential)
#   8. PT-SFT eval (4 tasks)
#
# Skip rules: each phase checks for its expected output file/dir; if present,
# the phase is skipped. Re-run is safe.
#
# Usage:
#   bash scripts/run_groupB.sh [phase_start] [phase_end]
# Default: phases 1-8.
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

PHASE_START="${1:-1}"
PHASE_END="${2:-8}"

LOG_DIR="logs/groupB"
mkdir -p "$LOG_DIR"

GPUS_HEAD=(1 6 7 4)              # one GPU per task for Stage-1 head training
DDP_GPUS_LORA="1,2,3,4"          # 4-GPU DDP for the rule-chaining Stage-2 LoRA
EVAL_GPU_SHARDS=(1 2 3 4 6 7)    # 6-way sharded eval

TORCHRUN=/data/yuyu/.local/bin/torchrun
export HYPPLAN_DIST_BACKEND=gloo

echo "Phase $PHASE_START..$PHASE_END"

# ---- Phase 0: pre-flight checks ----
for f in \
  data/rulechain_train.jsonl data/rulechain_val.jsonl data/rulechain_test.jsonl \
  data/synthlogic_train.jsonl data/synthlogic_test.jsonl \
  data/clutrr_train.jsonl data/clutrr_test.jsonl \
  data/rulechain_train_sft_plan.jsonl \
  data/synthlogic_train_sft_plan.jsonl \
  data/clutrr_train_sft_plan.jsonl \
do
  [ -f "$f" ] || { echo "MISSING data: $f"; exit 1; }
done
echo "[0/8] Data files present"

# ---- Phase 1: Stage-1 heads (parallel) ----
if [ "$PHASE_START" -le 1 ] && [ "$PHASE_END" -ge 1 ]; then
  echo "[1/8] Stage-1 heads"
  declare -A HEAD_PIDS
  TASKS=(rulechain synthlogic clutrr)
  for i in 0 1 2 3; do
    task=${TASKS[$i]}
    gpu=${GPUS_HEAD[$i]}
    out="checkpoints/head_${task}_qwen14b_rank/head.pt"
    if [ -f "$out" ]; then
      echo "  skip $task (head exists)"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu nohup \
      python -m src.train_head \
        --config configs/head_${task}_qwen14b_rank.yaml \
        > "$LOG_DIR/head_${task}.log" 2>&1 &
    HEAD_PIDS[$task]=$!
    echo "  $task on GPU $gpu (PID ${HEAD_PIDS[$task]})"
  done
  for task in "${!HEAD_PIDS[@]}"; do
    pid=${HEAD_PIDS[$task]}
    wait $pid && echo "  done: $task" || { echo "FAIL $task"; exit 1; }
  done
fi

# ---- Phase 2: rule-chaining Stage-2 LoRA (DDP 4-GPU) ----
if [ "$PHASE_START" -le 2 ] && [ "$PHASE_END" -ge 2 ]; then
  echo "[2/8] Rule-chaining Stage-2 LoRA"
  if [ -d "checkpoints/dagger_stage2_rulechain_bal_r4/lora" ]; then
    echo "  skip (LoRA exists)"
  else
    CUDA_VISIBLE_DEVICES=$DDP_GPUS_LORA \
      $TORCHRUN --nproc_per_node=4 --master_port=29501 \
        src/train_stage2_dagger_ood.py \
          --task rulechain \
          --config configs/stage2_dagger_rulechain_balanced.yaml \
          --use_z 1 \
        | tee "$LOG_DIR/stage2_rulechain.log"
  fi
fi

# ---- Phase 3: OOD 4-cell eval matrix ----
if [ "$PHASE_START" -le 3 ] && [ "$PHASE_END" -ge 3 ]; then
  echo "[3/8] OOD 4-cell eval matrix"
  CKPT="checkpoints/dagger_stage2_rulechain_bal_r4"
  for ood_task in synthlogic clutrr; do
    head_path="checkpoints/head_${ood_task}_qwen14b_rank/head.pt"
    test_data="data/${ood_task}_test.jsonl"
    out_base="results/eval_${ood_task}_v1"
    mkdir -p "$out_base"
    for mode in base lora lora_randz lora_taskz; do
      out="$out_base/${ood_task}_${mode}.jsonl"
      [ -f "$out" ] && { echo "  skip $ood_task/$mode"; continue; }
      echo "  eval $ood_task / $mode"
      for i in 0 1 2 3 4 5; do
        gpu=${EVAL_GPU_SHARDS[$i]}
        CUDA_VISIBLE_DEVICES=$gpu python3.10 -m src.eval_ood_generic \
          --mode $mode --ckpt_dir $CKPT \
          --head_path $head_path --task $ood_task \
          --test_data $test_data \
          --output ${out%.jsonl}_shard${i}.jsonl \
          --shard_rank $i --shard_world 6 \
          > "$LOG_DIR/eval_${ood_task}_${mode}_shard${i}.log" 2>&1 &
      done
      wait
      cat ${out%.jsonl}_shard*.jsonl > $out
      python -m src.score_ood --task $ood_task --input "$out" \
        | tee "$LOG_DIR/score_${ood_task}_${mode}.log"
    done
  done
fi

# ---- Phase 4: matched-prompt control on synthlogic ----
if [ "$PHASE_START" -le 4 ] && [ "$PHASE_END" -ge 4 ]; then
  echo "[4/8] Matched-prompt control on synthlogic"
  # TODO: requires an eval driver flag for CoT prompt + dense-z (not yet
  # exposed via eval_ood_generic.py). For Group A this was done in
  # eval_pq_dense_z. We'll write the analog (eval_synthlogic_dense_z.py)
  # before this phase runs. Skipping for now.
  echo "  (skipped — requires dense-z eval driver, separate PR)"
fi

# ---- Phase 5: in-domain Stage-2 LoRAs (sequential) ----
if [ "$PHASE_START" -le 5 ] && [ "$PHASE_END" -ge 5 ]; then
  echo "[5/8] In-domain Stage-2 LoRAs"
  for task in synthlogic clutrr; do
    out="checkpoints/dagger_stage2_${task}_indomain/lora"
    [ -d "$out" ] && { echo "  skip $task"; continue; }
    CUDA_VISIBLE_DEVICES=$DDP_GPUS_LORA \
      $TORCHRUN --nproc_per_node=4 --master_port=29502 \
        src/train_stage2_dagger_ood.py \
          --task $task \
          --config configs/stage2_dagger_${task}_qwen14b.yaml \
          --use_z 1 \
        | tee "$LOG_DIR/stage2_${task}_indomain.log"
  done
fi

# ---- Phase 6: in-domain HypPlan eval ----
if [ "$PHASE_START" -le 6 ] && [ "$PHASE_END" -ge 6 ]; then
  echo "[6/8] In-domain HypPlan eval"
  for task in synthlogic clutrr; do
    ckpt="checkpoints/dagger_stage2_${task}_indomain"
    head_path="checkpoints/head_${task}_qwen14b_rank/head.pt"
    test_data="data/${task}_test.jsonl"
    out="results/eval_stage2_indomain/${task}.jsonl"
    mkdir -p "$(dirname $out)"
    [ -f "$out" ] && { echo "  skip $task"; continue; }
    for i in 0 1 2 3 4 5; do
      gpu=${EVAL_GPU_SHARDS[$i]}
      CUDA_VISIBLE_DEVICES=$gpu python3.10 -m src.eval_stage2_indomain \
        --task $task \
        --ckpt_dir $ckpt --head_path $head_path \
        --test_data $test_data \
        --output ${out%.jsonl}_shard${i}.jsonl \
        --shard_rank $i --shard_world 6 \
        > "$LOG_DIR/eval_indomain_${task}_shard${i}.log" 2>&1 &
    done
    wait
    cat ${out%.jsonl}_shard*.jsonl > $out
    python -m src.score_ood --task $task --input "$out"
  done
fi

# ---- Phase 7: PT-SFT baselines ----
if [ "$PHASE_START" -le 7 ] && [ "$PHASE_END" -ge 7 ]; then
  echo "[7/8] PT-SFT baselines"
  for task in rulechain synthlogic clutrr; do
    out="checkpoints/sft_pt_${task}_qwen14b/lora"
    [ -d "$out" ] && { echo "  skip $task"; continue; }
    CUDA_VISIBLE_DEVICES=$DDP_GPUS_LORA \
      $TORCHRUN --nproc_per_node=4 --master_port=29503 \
        src/train_sft_24_qwen.py \
          --config configs/sft_pt_${task}_qwen14b.yaml \
        | tee "$LOG_DIR/sft_pt_${task}.log"
  done
fi

# ---- Phase 8: PT-SFT eval ----
if [ "$PHASE_START" -le 8 ] && [ "$PHASE_END" -ge 8 ]; then
  echo "[8/8] PT-SFT eval"
  for task in synthlogic clutrr; do
    test_data="data/${task}_test.jsonl"
    out="results/eval_pt_groupB/${task}.jsonl"
    mkdir -p "$(dirname $out)"
    [ -f "$out" ] && { echo "  skip $task"; continue; }
    for i in 0 1 2 3 4 5; do
      gpu=${EVAL_GPU_SHARDS[$i]}
      CUDA_VISIBLE_DEVICES=$gpu python3.10 -m src.eval_pt_ood \
        --task $task \
        --lora_adapter checkpoints/sft_pt_${task}_qwen14b/lora \
        --test_data $test_data \
        --output ${out%.jsonl}_shard${i}.jsonl \
        --shard_rank $i --shard_world 6 \
        > "$LOG_DIR/eval_pt_${task}_shard${i}.log" 2>&1 &
    done
    wait
    cat ${out%.jsonl}_shard*.jsonl > $out
    python -m src.score_ood --task $task --input "$out"
  done
fi

echo "DONE."
