"""Re-score Qwen BW/PQ/GC method results at first-100 records, after
the canonical limit for these tasks dropped from 200 to 100.

Outputs an ASCII table with old (n=200) vs new (n=100) accuracy so we
can update docs/results_summary.md.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable

from src.score_ood import (
    score_prontoqa,
    score_blocksworld,
    score_blocksworld_goal_reaching,
    score_rulechain,
    score_g24,
    score_proofwriter,
    score_clutrr,
)


def _idx(rid: str) -> int:
    try:
        return int(rid.rsplit("_", 1)[-1])
    except Exception:
        return -1


def _load_all(pattern: str) -> list[dict]:
    rs = []
    for f in sorted(glob.glob(pattern)):
        try:
            for line in open(f):
                rs.append(json.loads(line))
        except Exception as e:
            print(f"  SKIP {f}: {type(e).__name__}")
    by_id = {}
    for r in rs:
        rid = r.get("id")
        if rid is None:
            continue
        if rid not in by_id:
            by_id[rid] = r
    return sorted(by_id.values(), key=lambda r: _idx(str(r.get("id", ""))))


def _load_prompts(test_path: str) -> dict:
    out = {}
    for line in open(test_path):
        r = json.loads(line)
        if "id" in r:
            out[r["id"]] = r.get("prompt") or r.get("question") or ""
    return out


def _score_record(task: str, r: dict, gen_field: str = "generation") -> bool:
    gen = r.get(gen_field) or r.get("generation") or ""
    if task in ("pq", "prontoqa"):
        return score_prontoqa(gen, r.get("answer_label"))
    if task == "bw":
        # BW is goal-reaching for both PT-SFT and OVM
        ok, _ = score_blocksworld_goal_reaching(gen, r.get("prompt", ""))
        return ok
    if task == "gc":
        from src.oracle_graphcolor import Problem, parse_coloring, score_coloring
        p = Problem(n=r["n"], edges=tuple(map(tuple, r["edges"])))
        coloring = parse_coloring(gen, p)
        return score_coloring(p, coloring)
    raise ValueError(task)


def _tally(label: str, records: list[dict], task: str,
           is_already_scored: bool, gen_field: str = "generation",
           score_field: str = "correct") -> tuple[int, int, int, int]:
    """Returns (ok200, n200, ok100, n100)."""
    rs = records
    if is_already_scored:
        ok200 = sum(1 for r in rs[:200] if r.get(score_field))
        ok100 = sum(1 for r in rs[:100] if r.get(score_field))
    else:
        ok200 = sum(1 for r in rs[:200] if _score_record(task, r, gen_field))
        ok100 = sum(1 for r in rs[:100] if _score_record(task, r, gen_field))
    n200 = min(len(rs), 200)
    n100 = min(len(rs), 100)
    print(f"{label:35s} n200: {ok200:>3}/{n200} = {100*ok200/max(n200,1):5.1f}%   "
          f"n100: {ok100:>3}/{n100} = {100*ok100/max(n100,1):5.1f}%")
    return ok200, n200, ok100, n100


def main() -> None:
    print("=" * 80)
    print(" Qwen BW/PQ/GC re-tally at first-100 (after canonical change)")
    print("=" * 80)

    # Greedy / SC are already scored with top1_ok / majority_ok
    print("\n--- Greedy + SC (existing top1_ok / majority_ok) ---")
    fewshot_files = {
        "BW greedy":  ("results/missing/bw_greedy.jsonl", "top1_ok"),
        "BW SC=5":    ("results/baselines/bw_sc.jsonl",   "majority_ok"),
        "PQ greedy":  ("results/missing/pq_greedy.jsonl", "top1_ok"),
        "PQ SC=5":    ("results/baselines/pq_sc.jsonl",   "majority_ok"),
        "GC greedy":  ("results/missing/gc_greedy.jsonl", "top1_ok"),
        "GC SC=5":    ("results/baselines/gc_sc.jsonl",   "majority_ok"),
    }
    for name, (p, k) in fewshot_files.items():
        rs = [json.loads(l) for l in open(p)]
        rs.sort(key=lambda r: _idx(r.get("id", "")))
        ok200 = sum(1 for r in rs[:200] if r.get(k))
        ok100 = sum(1 for r in rs[:100] if r.get(k))
        print(f"{name:35s} n200: {ok200:>3}/{min(len(rs),200)}   "
              f"n100: {ok100:>3}/{min(len(rs),100)} = {100*ok100/max(min(len(rs),100),1):5.1f}%")

    # ToT BFS — has "top1_correct" field
    print("\n--- ToT BFS (top1_correct field) ---")
    for task in ["bw", "pq", "gc"]:
        rs = _load_all(f"results/tot_ood/{task}/{task}_shard*.jsonl")
        ok200 = sum(1 for r in rs[:200] if r.get("top1_correct"))
        ok100 = sum(1 for r in rs[:100] if r.get("top1_correct"))
        print(f"ToT-BFS {task.upper():4s}                       "
              f"n200: {ok200:>3}/{min(len(rs),200)} = {100*ok200/max(min(len(rs),200),1):5.1f}%   "
              f"n100: {ok100:>3}/{min(len(rs),100)} = {100*ok100/max(min(len(rs),100),1):5.1f}%")

    # PT-SFT — generation field, needs scoring
    print("\n--- PT-SFT (re-score with task scorer) ---")
    for task, pat in [("bw", "results/eval_pt_ood/bw_shard*.jsonl"),
                      ("pq", "results/eval_pt_ood/pq_shard*.jsonl")]:
        rs = _load_all(pat)
        if not rs:
            print(f"PT-SFT {task.upper()}: no files")
            continue
        _tally(f"PT-SFT {task.upper()}", rs, task,
               is_already_scored=False, gen_field="generation")
    # GC PT-SFT
    rs = _load_all("results/eval_gc_v1/gc_pt_sft.jsonl")
    if rs:
        _tally("PT-SFT GC", rs, "gc",
               is_already_scored=False, gen_field="generation")

    # OVM — ovm_generation field, needs scoring; BW also needs prompt
    print("\n--- OVM (re-score with task scorer) ---")
    test_paths = {
        "bw": "data/blocksworld_test.jsonl",
        "gc": "data/graphcolor_test.jsonl",
    }
    for task, pat in [("bw", "results/ovm/bw_ovm_shard*.jsonl"),
                      ("gc", "results/ovm/gc_ovm_v2_shard*.jsonl")]:
        rs = _load_all(pat)
        if task == "bw":
            prompts = _load_prompts(test_paths[task])
            for r in rs:
                if not r.get("prompt") and r.get("id") in prompts:
                    r["prompt"] = prompts[r["id"]]
        if task == "gc":
            # graphcolor scorer uses r["n"], r["edges"]; the OVM
            # records already have those.
            pass
        _tally(f"OVM {task.upper()}", rs, task,
               is_already_scored=False, gen_field="ovm_generation")

    # HypPlan in-domain
    print("\n--- HypPlan in-domain (correct field) ---")
    for task, pat in [("bw", "results/eval_stage2_indomain/bw/bw_shard*.jsonl"),
                      ("pq", "results/eval_stage2_indomain/pq/pq_shard*.jsonl"),
                      ("gc", "results/eval_stage2_indomain/gc/gc_shard*.jsonl")]:
        rs = _load_all(pat)
        ok200 = sum(1 for r in rs[:200] if r.get("correct"))
        ok100 = sum(1 for r in rs[:100] if r.get("correct"))
        print(f"HypPlan {task.upper():4s}                      "
              f"n200: {ok200:>3}/{min(len(rs),200)} = {100*ok200/max(min(len(rs),200),1):5.1f}%   "
              f"n100: {ok100:>3}/{min(len(rs),100)} = {100*ok100/max(min(len(rs),100),1):5.1f}%")


if __name__ == "__main__":
    main()
