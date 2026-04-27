"""Generate one solution trajectory per Countdown problem.

For each problem in data/cd_{split}.jsonl, pick the lex-first winning op at
each state and walk to the target. Writes data/cd_{split}_sft.jsonl with
{pool, target, problem_idx, text, step_offsets}.

Requires the offline oracle cache at data/cd_oracle_cache/{split}.pkl
(build via data/build_oracle_cache.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.oracle_cd import CountdownOracle, apply_step


def find_trajectory(pool: list[int], target: int, warm_cache: dict
                    ) -> list[tuple[int, str, int, int]] | None:
    """Walk from root to size-1 by picking the lex-first winning op at each step."""
    oracle = CountdownOracle(target)
    oracle._cache.update(warm_cache)

    state = tuple(sorted(pool))
    history: list[tuple[int, str, int, int]] = []
    while len(state) > 1:
        winners = oracle.winning_ops(state)
        if not winners:
            return None
        winners.sort(key=lambda w: (w[0], w[1], w[2]))
        sym, a, b, r = winners[0]
        history.append((a, sym, b, r))
        state = apply_step(state, a, b, r)
    return history


def trajectory_to_text(pool: list[int], target: int,
                       history: list[tuple[int, str, int, int]]) -> str:
    pool_str = " ".join(str(n) for n in sorted(pool))
    lines = [f"Problem: {pool_str} | Target: {target}"]
    working = list(sorted(pool))
    for i, (a, op, b, r) in enumerate(history):
        working.remove(a)
        working.remove(b)
        working.append(r)
        is_last = i == len(history) - 1
        if is_last:
            lines.append(f"Step {i+1}: {a} {op} {b} = {r}. Answer: {target}")
        else:
            rem_str = " ".join(str(x) for x in sorted(working))
            lines.append(f"Step {i+1}: {a} {op} {b} = {r}. Remaining: {rem_str}")
    return "\n".join(lines)


_STEP_RE = re.compile(r"^Step \d+:", re.MULTILINE)


def find_step_offsets(text: str) -> list[int]:
    return [m.start() for m in _STEP_RE.finditer(text)]


def process_split(jsonl_in: Path, cache_in: Path, jsonl_out: Path) -> None:
    with jsonl_in.open() as f:
        problems = [json.loads(line) for line in f]
    with cache_in.open("rb") as f:
        caches = pickle.load(f)

    failed = 0
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w") as f:
        for p, memo in zip(problems, caches):
            history = find_trajectory(p["pool"], p["target"], memo)
            if history is None:
                failed += 1
                continue
            text = trajectory_to_text(p["pool"], p["target"], history)
            f.write(json.dumps({
                "pool": p["pool"],
                "target": p["target"],
                "problem_idx": p["problem_idx"],
                "text": text,
                "step_offsets": find_step_offsets(text),
            }) + "\n")
    ok = len(problems) - failed
    print(f"  {jsonl_out}: {ok}/{len(problems)} trajectories "
          f"(failed: {failed})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--cache_dir", type=str, default="data/cd_oracle_cache")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    for split in ["train", "val", "test"]:
        jsonl_in = data_dir / f"cd_{split}.jsonl"
        cache_in = cache_dir / f"{split}.pkl"
        jsonl_out = data_dir / f"cd_{split}_sft.jsonl"
        if not (jsonl_in.exists() and cache_in.exists()):
            print(f"[{split}] skip (missing input)")
            continue
        print(f"[{split}]")
        process_split(jsonl_in, cache_in, jsonl_out)


if __name__ == "__main__":
    main()
