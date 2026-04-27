"""Generate number-path / reachability JSONL data (Group A OOD #1 candidate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_numpath import (
    Problem, format_gold_trajectory, format_question, generate_problem,
    render_state,
)


def problem_to_record(p: Problem, idx: int, split: str, depth: int) -> dict:
    return {
        "id": f"numpath_{split}_{idx}",
        "start": p.start,
        "target": p.target,
        "ops": [{"kind": o.kind, "const": o.const} for o in p.ops],
        "max_value": p.max_value,
        "n_steps": depth,
        "prompt": format_question(p),
        "init_state_text": render_state(p, p.start),
        "answer_label": format_gold_trajectory(p),
        "split": split,
    }


def gen_split(n: int, depths: list[int], seed_offset: int, split: str,
                op_set_size: int) -> list[dict]:
    out: list[dict] = []
    di = 0
    seed = seed_offset
    fail = 0
    while len(out) < n and fail < 1000:
        depth = depths[di % len(depths)]
        di += 1
        try:
            p = generate_problem(target_depth=depth, op_set_size=op_set_size,
                                  seed=seed)
            out.append(problem_to_record(p, len(out), split, depth))
            fail = 0
        except RuntimeError:
            fail += 1
        seed += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--depths", default="3,4,5")
    ap.add_argument("--op_set_size", type=int, default=4)
    ap.add_argument("--seed_base", type=int, default=2024)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, n, seed in [
        ("train", args.n_train, args.seed_base),
        ("val", args.n_val, args.seed_base + 100_000),
        ("test", args.n_test, args.seed_base + 200_000),
    ]:
        if n <= 0:
            continue
        print(f"Generating numpath/{split}: n={n}, depths={depths}, "
              f"op_set_size={args.op_set_size}", flush=True)
        recs = gen_split(n=n, depths=depths, seed_offset=seed, split=split,
                          op_set_size=args.op_set_size)
        path = out_dir / f"numpath_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
