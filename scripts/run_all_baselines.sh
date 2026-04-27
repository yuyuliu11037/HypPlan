#!/usr/bin/env bash
# Run the full ToT + SC + PT-SFT-eval baseline grid for both groups,
# committing + pushing after each (task, mode) cell.
#
# ToT (5 new tasks since v1 already has PQ/BW/GC):
#   g24, numpath, rulechain, clutrr, proofwriter
# SC (all 8 tasks):
#   g24, numpath, bw, gc, rulechain, pq, clutrr, proofwriter
# PT-SFT eval (4 new — requires the LoRA to be trained first):
#   numpath, rulechain, clutrr, proofwriter
#
# Skips a cell if the result file already exists.
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

# =============================================================
# ToT block (top-1 + any-of-5)
# =============================================================
TOT_TASKS=(
  "numpath data/numpath_test.jsonl"
  "rulechain data/rulechain_test.jsonl"
  "clutrr data/clutrr_test.jsonl"
  "proofwriter data/proofwriter_test.jsonl"
  "g24 data/24_test.jsonl --limit 100"
)

for entry in "${TOT_TASKS[@]}"; do
  read -r task data extra <<<"$entry"
  out="results/baselines/${task}_tot.jsonl"
  if [ -s "$out" ]; then
    echo "skip $task tot (exists)"
    continue
  fi
  bash scripts/run_one_baseline.sh "$task" tot 6 "$data" $extra
done

# =============================================================
# SC block (majority vote)
# =============================================================
SC_TASKS=(
  "numpath data/numpath_test.jsonl"
  "rulechain data/rulechain_test.jsonl"
  "clutrr data/clutrr_test.jsonl"
  "proofwriter data/proofwriter_test.jsonl"
  "pq data/prontoqa_test.jsonl"
  "bw data/blocksworld_test.jsonl"
  "gc data/graphcolor_test.jsonl"
  "g24 data/24_test.jsonl --limit 100"
)

for entry in "${SC_TASKS[@]}"; do
  read -r task data extra <<<"$entry"
  out="results/baselines/${task}_sc.jsonl"
  if [ -s "$out" ]; then
    echo "skip $task sc (exists)"
    continue
  fi
  bash scripts/run_one_baseline.sh "$task" sc 6 "$data" $extra
done

echo "All ToT + SC baselines done. PT-SFT trainings + evals are run via a
separate script (scripts/run_pt_sft_pipeline.sh) once trainings finish."
