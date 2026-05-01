"""Phase 3 launcher — PT-SFT training + eval for gpt-oss-20b and
mistral-24b across 8 tasks.

For each (base, task) cell:
  1. If checkpoint LoRA doesn't exist, train (single-GPU per cell).
  2. Eval with src.eval_pt_ood, score with src.score_ood.

Cells run 1 GPU per cell, 8 in parallel; remaining queued.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


# (base_tag, base_model, task_key, ckpt_tag, config_path,
#  test_data, limit, score_task)
CELLS = [
    # gpt-oss-20b: 24 already trained, others to train
    ("gptoss20b", "openai/gpt-oss-20b", "g24", "24",
     "configs/gptoss/sft_pt_24_gptoss20b.yaml",
     "data/24_test.jsonl", 100, "g24"),
    ("gptoss20b", "openai/gpt-oss-20b", "pq", "pq",
     "configs/gptoss/sft_pt_pq_gptoss20b.yaml",
     "data/prontoqa_test.jsonl", 100, "pq"),
    ("gptoss20b", "openai/gpt-oss-20b", "bw", "bw",
     "configs/gptoss/sft_pt_bw_gptoss20b.yaml",
     "data/blocksworld_test.jsonl", 100, "bw"),
    ("gptoss20b", "openai/gpt-oss-20b", "gc", "gc",
     "configs/gptoss/sft_pt_gc_gptoss20b.yaml",
     "data/graphcolor_test.jsonl", 100, "gc"),
    ("gptoss20b", "openai/gpt-oss-20b", "rulechain", "rulechain",
     "configs/gptoss/sft_pt_rulechain_gptoss20b.yaml",
     "data/rulechain_test.jsonl", 200, "rulechain"),
    ("gptoss20b", "openai/gpt-oss-20b", "clutrr", "clutrr_graph_v5",
     "configs/gptoss/sft_pt_clutrr_graph_v5_gptoss20b.yaml",
     "data/clutrr_graph_v5_test.jsonl", 200, "clutrr"),
    ("gptoss20b", "openai/gpt-oss-20b", "proofwriter", "proofwriter",
     "configs/gptoss/sft_pt_proofwriter_gptoss20b.yaml",
     "data/proofwriter_test.jsonl", 200, "proofwriter"),
    ("gptoss20b", "openai/gpt-oss-20b", "nqueens", "nqueens",
     "configs/gptoss/sft_pt_nqueens_gptoss20b.yaml",
     "data/nqueens_test.jsonl", 45, "nqueens"),
    # mistral-24b: 8 to train
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "g24", "24",
     "configs/mistral/sft_pt_24_mistral24b.yaml",
     "data/24_test.jsonl", 100, "g24"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "pq", "pq",
     "configs/mistral/sft_pt_pq_mistral24b.yaml",
     "data/prontoqa_test.jsonl", 100, "pq"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "bw", "bw",
     "configs/mistral/sft_pt_bw_mistral24b.yaml",
     "data/blocksworld_test.jsonl", 100, "bw"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "gc", "gc",
     "configs/mistral/sft_pt_gc_mistral24b.yaml",
     "data/graphcolor_test.jsonl", 100, "gc"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "rulechain", "rulechain",
     "configs/mistral/sft_pt_rulechain_mistral24b.yaml",
     "data/rulechain_test.jsonl", 200, "rulechain"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "clutrr", "clutrr_graph_v5",
     "configs/mistral/sft_pt_clutrr_graph_v5_mistral24b.yaml",
     "data/clutrr_graph_v5_test.jsonl", 200, "clutrr"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "proofwriter", "proofwriter",
     "configs/mistral/sft_pt_proofwriter_mistral24b.yaml",
     "data/proofwriter_test.jsonl", 200, "proofwriter"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "nqueens", "nqueens",
     "configs/mistral/sft_pt_nqueens_mistral24b.yaml",
     "data/nqueens_test.jsonl", 45, "nqueens"),
]

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
LOG_DIR = Path("logs/phase3_ptsft")
OUT_DIR = Path("results/phase3_ptsft")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def lora_dir(base_tag: str, ckpt_tag: str) -> Path:
    return Path(f"checkpoints/sft_pt_{ckpt_tag}_{base_tag}/lora")


def out_path(base_tag: str, ckpt_tag: str) -> Path:
    return OUT_DIR / f"{base_tag}_{ckpt_tag}.jsonl"


def cell_done(cell) -> bool:
    base_tag, _, _, ckpt_tag, _, _, limit, _ = cell
    p = out_path(base_tag, ckpt_tag)
    if not p.exists():
        return False
    try:
        return sum(1 for _ in open(p)) >= limit
    except Exception:
        return False


def launch_cell(gpu: int, cell) -> subprocess.Popen:
    base_tag, model, task_key, ckpt_tag, cfg_path, test_data, limit, score_task = cell
    ld = lora_dir(base_tag, ckpt_tag)
    op = out_path(base_tag, ckpt_tag)
    log_path = LOG_DIR / f"{base_tag}_{ckpt_tag}.log"
    train_then_eval = f"""
set -e
set -o pipefail
echo "=== START $(date -Iseconds) ==="
if [ ! -d "{ld}" ]; then
  echo "=== TRAIN PT-SFT {base_tag} :: {ckpt_tag} ==="
  python3.10 -m src.train_sft_pt_qwen --config "{cfg_path}"
else
  echo "[skip-train] LoRA exists at {ld}"
fi
echo "=== EVAL PT-SFT {base_tag} :: {ckpt_tag} ==="
python3.10 -m src.eval_pt_ood \\
  --task "{task_key}" \\
  --base_model "{model}" \\
  --lora_adapter "{ld}" \\
  --test_data "{test_data}" \\
  --output "{op}" \\
  --max_new_tokens 384 \\
  --limit {limit}
echo "=== SCORE PT-SFT {base_tag} :: {ckpt_tag} ==="
python3.10 -m src.score_ood --input "{op}" --task "{score_task}"
echo "=== DONE $(date -Iseconds) ==="
""".strip()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    fout = open(log_path, "w")
    print(f"[GPU{gpu}] LAUNCH PT-SFT {base_tag} :: {ckpt_tag}", flush=True)
    p = subprocess.Popen(["bash", "-c", train_then_eval],
                          env=env, stdout=fout, stderr=subprocess.STDOUT)
    p.gpu = gpu
    p.cell = cell
    p.log_path = log_path
    p.fout = fout
    p.t0 = time.time()
    return p


def main() -> None:
    queue = [c for c in CELLS if not cell_done(c)]
    if not queue:
        print("All Phase-3 cells already done.")
        return
    print(f"Phase 3: {len(queue)} cells to run")
    for c in queue:
        ld = lora_dir(c[0], c[3])
        ld_state = "✓" if ld.exists() else "✗"
        print(f"  - {c[0]} :: {c[3]}  (lora={ld_state} task={c[2]} n={c[6]})")

    free = list(GPUS)
    running: list[subprocess.Popen] = []
    while queue or running:
        while free and queue:
            gpu = free.pop(0)
            cell = queue.pop(0)
            running.append(launch_cell(gpu, cell))
        if running:
            done = []
            while not done:
                time.sleep(30)
                for p in running:
                    if p.poll() is not None:
                        done.append(p)
            for p in done:
                running.remove(p)
                p.fout.close()
                dur = time.time() - p.t0
                tag = f"{p.cell[0]}::{p.cell[3]}"
                print(f"[GPU{p.gpu}] DONE {tag} rc={p.returncode} "
                      f"{dur/60:.1f}m", flush=True)
                if p.returncode != 0:
                    try:
                        with open(p.log_path) as f:
                            for line in f.readlines()[-25:]:
                                print(f"    {line.rstrip()}", flush=True)
                    except Exception:
                        pass
                free.append(p.gpu)
    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
