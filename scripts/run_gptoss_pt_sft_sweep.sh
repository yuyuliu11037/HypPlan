#!/usr/bin/env bash
# GPT-OSS-20B PT-SFT sweep across 8 datasets on a single GPU.
# Per task: train SFT-PT LoRA → run eval → save jsonl + summary → commit.
#
# Skip cells whose final jsonl already exists.
set -e
set -o pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

GPU=${GPU:-2}
mkdir -p logs/gptoss_ptsft results/eval_pt_ood/gptoss20b

# task | sft_data_tag | test_data | record_limit
TASKS=(
  "24:24:data/24_test.jsonl:100"
  "pq:prontoqa:data/prontoqa_test.jsonl:100"
  "bw:blocksworld:data/blocksworld_test.jsonl:100"
  "gc:graphcolor:data/graphcolor_test.jsonl:100"
  "rulechain:rulechain:data/rulechain_test.jsonl:200"
  "clutrr:clutrr:data/clutrr_test.jsonl:200"
  "numpath:numpath:data/numpath_test.jsonl:200"
  "proofwriter:proofwriter:data/proofwriter_test.jsonl:200"
)

run_one() {
  local task=$1 data_tag=$2 test_data=$3 limit=$4
  local cfg="configs/gptoss/sft_pt_${task}_gptoss20b.yaml"
  local lora_dir="checkpoints/sft_pt_${task}_gptoss20b/lora"
  local out="results/eval_pt_ood/gptoss20b/${task}.jsonl"
  local train_log="logs/gptoss_ptsft/train_${task}.log"
  local eval_log="logs/gptoss_ptsft/eval_${task}.log"

  if [ -s "$out" ] && [ "$(wc -l < "$out")" -ge "$limit" ]; then
    echo "[skip] $task (eval done: $(wc -l < "$out")/$limit)"
    return
  fi

  if [ ! -d "$lora_dir" ]; then
    echo "=== train PT-SFT GPT-OSS-20B :: $task ==="
    CUDA_VISIBLE_DEVICES=$GPU python3.10 -m src.train_sft_pt_qwen \
      --config "$cfg" > "$train_log" 2>&1
    tail -3 "$train_log"
  else
    echo "[skip] $task LoRA exists"
  fi

  echo "=== eval PT-SFT GPT-OSS-20B :: $task ==="
  CUDA_VISIBLE_DEVICES=$GPU python3.10 -m src.eval_pt_ood \
    --task "$task" --base_model openai/gpt-oss-20b \
    --lora_adapter "$lora_dir" \
    --test_data "$test_data" \
    --output "$out" \
    --max_new_tokens 384 \
    --limit "$limit" \
    > "$eval_log" 2>&1
  tail -3 "$eval_log"

  # Score
  local summary="results/eval_pt_ood/gptoss20b/${task}_summary.txt"
  python3.10 - <<PY > "$summary"
import json
recs = [json.loads(l) for l in open("$out")]
n = len(recs)
print(f"task=$task gpt-oss-20b PT-SFT n={n}")
PY
  cat "$summary"

  git add -f "$out" "$summary" "$lora_dir" 2>/dev/null || true
  git commit -m "gpt-oss-20b PT-SFT: $task eval ($(wc -l < $out)/$limit records)" 2>&1 | tail -1 || true
  git push origin main 2>&1 | tail -1 || true
}

echo "=== GPT-OSS-20B PT-SFT sweep on GPU $GPU ==="
for entry in "${TASKS[@]}"; do
  IFS=':' read -r task data_tag test_data limit <<<"$entry"
  run_one "$task" "$data_tag" "$test_data" "$limit"
done

echo "=== sweep done ==="
