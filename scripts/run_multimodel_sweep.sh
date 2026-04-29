#!/usr/bin/env bash
# Phase A: fewshot (greedy) + SC sweep for two new base models on
# all 8 datasets, single GPU (1), batch=1, isolated for clean
# latency / token-count measurements.
#
# Per cell: results/multimodel/{tag}_{task}_{mode}.jsonl
#          logs/multimodel/{tag}_{task}_{mode}.log
# Commits + pushes per cell. Skips cells whose jsonl already exists
# with the right number of lines.
set -e
set -o pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

GPU=${GPU:-1}
mkdir -p results/multimodel logs/multimodel

# Task table: name | data_file | record_limit
TASKS=(
  "g24:data/24_test.jsonl:100"
  "pq:data/prontoqa_test.jsonl:100"
  "bw:data/blocksworld_test.jsonl:100"
  "gc:data/graphcolor_test.jsonl:100"
  "rulechain:data/rulechain_test.jsonl:200"
  "clutrr:data/clutrr_test.jsonl:200"
  "numpath:data/numpath_test.jsonl:200"
  "proofwriter:data/proofwriter_test.jsonl:200"
)

run_cell() {
  local tag=$1 model=$2 task=$3 data=$4 limit=$5 mode=$6
  local out="results/multimodel/${tag}_${task}_${mode}.jsonl"
  local log="logs/multimodel/${tag}_${task}_${mode}.log"
  if [ -s "$out" ] && [ "$(wc -l < "$out")" -ge "$limit" ]; then
    echo "[skip] $tag $task $mode ($(wc -l < "$out")/$limit)"
    return
  fi
  echo "=== $tag $task $mode (limit=$limit) ==="
  CUDA_VISIBLE_DEVICES=$GPU python3.10 -m src.eval_baseline_kpath \
    --task "$task" --mode "$mode" \
    --base_model "$model" \
    --test_data "$data" \
    --out_path "$out" \
    --K 5 --temperature 0.7 --max_new_tokens 384 \
    --limit "$limit" \
    > "$log" 2>&1
  # quick summary
  python3.10 - <<PY
import json
recs = [json.loads(l) for l in open("$out")]
n = len(recs)
top1 = sum(1 for r in recs if r.get("top1_ok"))
maj = sum(1 for r in recs if r.get("majority_ok"))
toks = sum(int(r.get("n_gen_tokens") or 0) for r in recs)
lat  = sum(float(r.get("latency_s") or 0) for r in recs)
print(f"$tag $task $mode n={n}", flush=True)
if "$mode" == "greedy":
    print(f"  top1={top1}/{n}={top1/n:.0%}  tokens={toks}  latency={lat:.0f}s")
else:
    print(f"  maj={maj}/{n}={maj/n:.0%}  tokens={toks}  latency={lat:.0f}s")
PY
  git add -f "$out" "$log" 2>/dev/null || true
  git commit -m "multimodel sweep: $tag $task $mode" 2>&1 | tail -1 || true
  git push origin main 2>&1 | tail -1 || true
}

# Phase A1: GPT-OSS-20B
TAG_GPTOSS="gptoss20b"
MODEL_GPTOSS="openai/gpt-oss-20b"
echo "=== Phase A1: $MODEL_GPTOSS on GPU $GPU ==="
for entry in "${TASKS[@]}"; do
  IFS=':' read -r task data limit <<<"$entry"
  for mode in greedy sc; do
    run_cell "$TAG_GPTOSS" "$MODEL_GPTOSS" "$task" "$data" "$limit" "$mode"
  done
done

# Phase A2: Mistral-Small-3.2-24B
TAG_MISTRAL="mistral24b"
MODEL_MISTRAL="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
echo "=== Phase A2: $MODEL_MISTRAL on GPU $GPU ==="
for entry in "${TASKS[@]}"; do
  IFS=':' read -r task data limit <<<"$entry"
  for mode in greedy sc; do
    run_cell "$TAG_MISTRAL" "$MODEL_MISTRAL" "$task" "$data" "$limit" "$mode"
  done
done

echo "=== multimodel sweep done ==="
