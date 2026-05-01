"""Phase 2 launcher — ToT BFS on gpt-oss-20b and mistral-24b across 8 tasks.

ToT-BFS for non-G24 tasks uses src.tot_ood; G24 uses src.tot_baseline.
Fans out cells across 8 GPUs, 1 GPU per cell.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


# (base_tag, base_model, task, data_file, limit, out_name, use_4bit)
CELLS = []
for base_tag, base_model, use_4bit in [
    ("gptoss20b", "openai/gpt-oss-20b", 0),  # gpt-oss ships mxfp4
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", 1),
]:
    for task, data_file, limit in [
        ("clutrr",      "data/clutrr_graph_v5_test.jsonl",  200),
        ("proofwriter", "data/proofwriter_test.jsonl",      200),
        ("nqueens",     "data/nqueens_test.jsonl",           45),
        ("pq",          "data/prontoqa_test.jsonl",         100),
        ("bw",          "data/blocksworld_test.jsonl",      100),
        ("gc",          "data/graphcolor_test.jsonl",       100),
        ("rulechain",   "data/rulechain_test.jsonl",        200),
        # G24 uses tot_baseline.py — handled separately below
    ]:
        out_name = f"{base_tag}_{task}_tot"
        CELLS.append((base_tag, base_model, task, data_file, limit,
                      out_name, use_4bit))

# G24 cells (use tot_baseline.py)
for base_tag, base_model, use_4bit in [
    ("gptoss20b", "openai/gpt-oss-20b", 0),
    ("mistral24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", 1),
]:
    CELLS.append((base_tag, base_model, "g24",
                  "data/24_test_tot.jsonl", 100,
                  f"{base_tag}_g24_tot", use_4bit))

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
LOG_DIR = Path("logs/multimodel_tot")
OUT_DIR = Path("results/multimodel_tot")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def output_path(out_name: str, task: str) -> Path:
    return OUT_DIR / f"{out_name}.jsonl"


def already_done(out_path: Path, limit: int) -> bool:
    if not out_path.exists():
        return False
    try:
        return sum(1 for _ in open(out_path)) >= limit
    except Exception:
        return False


def launch(gpu: int, cell) -> subprocess.Popen:
    base_tag, model, task, data, limit, out_name, use_4bit = cell
    out_path = output_path(out_name, task)
    log_path = LOG_DIR / f"{out_name}.log"
    if task == "g24":
        # tot_baseline.py: --generator, --shared_model, --use_chat_template,
        # --test_data, --output_dir, --limit, --load_in_4bit
        cmd = [
            "python3.10", "-m", "src.tot_baseline",
            "--generator", model,
            "--shared_model",
            "--use_chat_template",
            "--test_data", data,
            "--output_dir", str(out_path.with_suffix("")),
            "--limit", str(limit),
            "--n_generate", "1", "--n_evaluate", "3", "--n_select", "5",
        ]
        if use_4bit:
            cmd.append("--load_in_4bit")
    else:
        cmd = [
            "python3.10", "-m", "src.tot_ood",
            "--task", task, "--model", model,
            "--test_data", data,
            "--output", str(out_path),
            "--limit", str(limit),
            "--use_4bit", str(int(use_4bit)),
        ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    fout = open(log_path, "w")
    print(f"[GPU{gpu}] LAUNCH {out_name} task={task} n={limit}", flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT)
    p.gpu = gpu
    p.cell = cell
    p.log_path = log_path
    p.out_path = out_path
    p.fout = fout
    p.t0 = time.time()
    return p


def main() -> None:
    queue = [c for c in CELLS
             if not already_done(output_path(c[5], c[2]), c[4])]
    if not queue:
        print("All Phase-2 cells already done.")
        return
    print(f"Phase 2: {len(queue)} cells to run")
    for c in queue:
        print(f"  - {c[5]} (task={c[2]} n={c[4]} model={c[1]})")

    free = list(GPUS)
    running: list[subprocess.Popen] = []
    while queue or running:
        while free and queue:
            gpu = free.pop(0)
            cell = queue.pop(0)
            running.append(launch(gpu, cell))
        if running:
            done = []
            while not done:
                time.sleep(20)
                for p in running:
                    if p.poll() is not None:
                        done.append(p)
            for p in done:
                running.remove(p)
                p.fout.close()
                dur = time.time() - p.t0
                print(f"[GPU{p.gpu}] DONE {p.cell[5]} rc={p.returncode} "
                      f"{dur/60:.1f}m", flush=True)
                if p.returncode != 0:
                    try:
                        with open(p.log_path) as f:
                            for line in f.readlines()[-20:]:
                                print(f"    {line.rstrip()}", flush=True)
                    except Exception:
                        pass
                free.append(p.gpu)
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
