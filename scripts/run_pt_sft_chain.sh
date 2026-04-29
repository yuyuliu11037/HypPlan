#!/usr/bin/env bash
# Run PT-SFT pipeline (train + 6-way sharded eval + commit + push) for
# all 4 new tasks sequentially. Each task: ~2 hr training + ~5 min eval.
# Total wall time: ~8-9 hr.
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

NEW_PT_TASKS=(numpath rulechain clutrr proofwriter)
DATA_OVERRIDES=(
  "data/numpath_test.jsonl"
  "data/rulechain_test.jsonl"
  "data/clutrr_test.jsonl"
  "data/proofwriter_test.jsonl"
)

for i in 0 1 2 3; do
  task=${NEW_PT_TASKS[$i]}
  data=${DATA_OVERRIDES[$i]}
  out="results/baselines/${task}_ptsft.jsonl"
  if [ -s "$out" ]; then
    echo "skip $task PT-SFT (result exists)"
    continue
  fi
  bash scripts/run_pt_sft_pipeline.sh "$task" "$data"
done

echo "All PT-SFT trainings + evals done."
