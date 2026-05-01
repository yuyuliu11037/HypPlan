"""Phase 1B launcher — fill 12 missing baseline cells (Greedy / SC) for
gpt-oss-20b and mistral-24b.

Runs 1 GPU per cell, fans out across all 8 GPUs, queues the rest.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


CELLS = [
    # (base_tag, base_model, task, data_file, limit, mode, out_name)
    # ---- gpt-oss-20b: 4 new cells ----
    ("gptoss20b", "openai/gpt-oss-20b", "clutrr",
     "data/clutrr_graph_v5_test.jsonl", 200, "greedy", "gptoss20b_clutrr_v5_greedy"),
    ("gptoss20b", "openai/gpt-oss-20b", "clutrr",
     "data/clutrr_graph_v5_test.jsonl", 200, "sc", "gptoss20b_clutrr_v5_sc"),
    ("gptoss20b", "openai/gpt-oss-20b", "nqueens",
     "data/nqueens_test.jsonl", 45, "greedy", "gptoss20b_nqueens_greedy"),
    ("gptoss20b", "openai/gpt-oss-20b", "nqueens",
     "data/nqueens_test.jsonl", 45, "sc", "gptoss20b_nqueens_sc"),
    # ---- mistral-24b: 8 new/redo cells ----
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "clutrr",
     "data/clutrr_graph_v5_test.jsonl", 200, "greedy", "mistral24b_clutrr_v5_greedy"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "clutrr",
     "data/clutrr_graph_v5_test.jsonl", 200, "sc", "mistral24b_clutrr_v5_sc"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "proofwriter",
     "data/proofwriter_test.jsonl", 200, "greedy", "mistral24b_proofwriter_greedy"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "proofwriter",
     "data/proofwriter_test.jsonl", 200, "sc", "mistral24b_proofwriter_sc"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "nqueens",
     "data/nqueens_test.jsonl", 45, "greedy", "mistral24b_nqueens_greedy"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "nqueens",
     "data/nqueens_test.jsonl", 45, "sc", "mistral24b_nqueens_sc"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "pq",
     "data/prontoqa_test.jsonl", 100, "sc", "mistral24b_pq_sc_v2"),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "rulechain",
     "data/rulechain_test.jsonl", 200, "sc", "mistral24b_rulechain_sc_v2"),
]

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
LOG_DIR = Path("logs/multimodel")
OUT_DIR = Path("results/multimodel")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def already_done(out_path: Path, limit: int) -> bool:
    if not out_path.exists():
        return False
    try:
        n = sum(1 for _ in open(out_path))
    except Exception:
        return False
    return n >= limit


def launch(gpu: int, cell) -> subprocess.Popen:
    base_tag, model, task, data, limit, mode, out_name = cell
    out_path = OUT_DIR / f"{out_name}.jsonl"
    log_path = LOG_DIR / f"{out_name}.log"
    cmd = [
        "python3.10", "-m", "src.eval_baseline_kpath",
        "--task", task, "--mode", mode,
        "--base_model", model,
        "--test_data", data,
        "--out_path", str(out_path),
        "--K", "5", "--temperature", "0.7", "--max_new_tokens", "384",
        "--limit", str(limit),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    fout = open(log_path, "w")
    print(f"[GPU{gpu}] LAUNCH {out_name}  ({task} {mode} n={limit})", flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT)
    p.gpu = gpu
    p.cell = cell
    p.log_path = log_path
    p.out_path = out_path
    p.fout = fout
    p.t0 = time.time()
    return p


def main() -> None:
    queue = list(CELLS)
    # Skip cells whose output already exists with the right size
    queue = [c for c in queue if not already_done(OUT_DIR / f"{c[6]}.jsonl", c[4])]
    if not queue:
        print("All cells already done.")
        return
    print(f"To run: {len(queue)} cells")
    for c in queue:
        print(f"  - {c[6]} ({c[2]} {c[5]} n={c[4]} on {c[1]})")

    free_gpus = list(GPUS)
    running: list[subprocess.Popen] = []
    while queue or running:
        # Fill empty GPU slots
        while free_gpus and queue:
            gpu = free_gpus.pop(0)
            cell = queue.pop(0)
            running.append(launch(gpu, cell))
        # Wait for any to finish
        if running:
            done = []
            while not done:
                time.sleep(15)
                for p in running:
                    if p.poll() is not None:
                        done.append(p)
            for p in done:
                running.remove(p)
                p.fout.close()
                dur = time.time() - p.t0
                tag = p.cell[6]
                ok = (p.returncode == 0)
                size = (p.out_path.stat().st_size if p.out_path.exists() else 0)
                print(f"[GPU{p.gpu}] DONE {tag} rc={p.returncode} "
                      f"{dur/60:.1f}m  size={size}", flush=True)
                if not ok:
                    print(f"[GPU{p.gpu}] tail of log:", flush=True)
                    try:
                        with open(p.log_path) as f:
                            for line in f.readlines()[-20:]:
                                print(f"    {line.rstrip()}", flush=True)
                    except Exception:
                        pass
                free_gpus.append(p.gpu)
    print("All Phase-1B cells finished.")


if __name__ == "__main__":
    main()
