"""Generate small-scale Countdown problems for Group A OOD #1 (lineq replacement).

Difficulty parameters:
- pool_size: number of operands (default 4, vs full Countdown's 6).
- pool_range: each operand sampled from [1, pool_max] (default 15).
- target_min, target_max: target sampled from [10, 99] (2-digit).

Each generated problem is guaranteed reachable: we sample a pool, run the
existing oracle to enumerate ALL reachable values from any subset, then pick
a target uniformly from values in the [target_min, target_max] range.

Schema per line:
  {
    "id": str,
    "pool": list[int],
    "target": int,
    "n_steps": int,                 # = pool_size - 1, all numbers consumed
    "prompt": str,                  # NL question with pool + target
    "init_state_text": str,         # for Stage-1 head input
    "answer_label": str,            # gold trajectory walked via oracle
  }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_cd import CountdownOracle, OPS, apply_step


def _sorted(xs: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(xs))


def reachable_targets(pool: tuple[int, ...]) -> set[int]:
    """All single-integer values reachable from `pool` by applying ops until
    one number remains. Capped at 5000 distinct sub-states to bound runtime."""
    pool = _sorted(pool)
    seen: dict[tuple[int, ...], None] = {pool: None}
    queue = [pool]
    finals: set[int] = set()
    while queue and len(seen) < 5000:
        state = queue.pop()
        if len(state) == 1:
            finals.add(state[0])
            continue
        n = len(state)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = state[i], state[j]
                rest = tuple(state[k] for k in range(n) if k != i and k != j)
                for sym, fn, comm in OPS:
                    if comm and a > b:
                        continue
                    r = fn(a, b)
                    if r is None:
                        continue
                    if r < 0 or r > 999:
                        continue
                    new_state = _sorted(rest + (r,))
                    if new_state in seen:
                        continue
                    seen[new_state] = None
                    queue.append(new_state)
    return finals


def gold_trajectory(pool: tuple[int, ...], target: int) -> list[tuple[int, str, int, int]]:
    """Return one shortest solving sequence as [(a, op, b, result), ...]."""
    oracle = CountdownOracle(target)
    state = _sorted(pool)
    hist: list[tuple[int, str, int, int]] = []
    while len(state) > 1:
        wins = oracle.winning_ops(state)
        if not wins:
            return []
        wins.sort(key=lambda w: (w[0], w[1], w[2]))
        sym, a, b, r = wins[0]
        hist.append((a, sym, b, r))
        state = apply_step(state, a, b, r)
    return hist


def render_prompt(pool: list[int], target: int) -> str:
    return (
        f"Use each number exactly once and the four arithmetic operations "
        f"(+, -, *, /) to make the target.\n"
        f"Numbers: {', '.join(str(n) for n in pool)}\n"
        f"Target: {target}"
    )


def render_init_state_text(pool: list[int], target: int) -> str:
    return (
        f"Numbers: {', '.join(str(n) for n in pool)}; Target: {target}"
    )


def render_gold(hist: list[tuple[int, str, int, int]], target: int) -> str:
    lines = []
    for i, (a, sym, b, r) in enumerate(hist):
        lines.append(f"Step {i + 1}: {a} {sym} {b} = {r}")
    lines.append(f"Answer: {target}")
    return "\n".join(lines)


def generate_problem(pool_size: int, pool_max: int, target_min: int,
                      target_max: int, rng: random.Random,
                      max_attempts: int = 50):
    for _ in range(max_attempts):
        pool = tuple(rng.choice(range(1, pool_max + 1)) for _ in range(pool_size))
        if len(set(pool)) < 2:  # avoid all-same trivial pools
            continue
        finals = reachable_targets(pool)
        candidates = [t for t in finals if target_min <= t <= target_max]
        if not candidates:
            continue
        target = rng.choice(candidates)
        hist = gold_trajectory(pool, target)
        if not hist:
            continue
        return list(pool), int(target), hist
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--name_prefix", default="cd_small")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--pool_size", type=int, default=4)
    ap.add_argument("--pool_max", type=int, default=15)
    ap.add_argument("--target_min", type=int, default=10)
    ap.add_argument("--target_max", type=int, default=99)
    ap.add_argument("--seed_base", type=int, default=3030)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.n_train, args.seed_base),
        ("val", args.n_val, args.seed_base + 100_000),
        ("test", args.n_test, args.seed_base + 200_000),
    ]
    for split, n, seed in splits:
        if n <= 0:
            continue
        print(f"Generating {args.name_prefix}/{split}: n={n}, "
              f"pool_size={args.pool_size}, pool_max={args.pool_max}, "
              f"target=[{args.target_min},{args.target_max}]", flush=True)
        rng = random.Random(seed)
        recs = []
        attempts = 0
        while len(recs) < n and attempts < n * 50:
            attempts += 1
            res = generate_problem(
                args.pool_size, args.pool_max, args.target_min, args.target_max,
                rng,
            )
            if res is None:
                continue
            pool, target, hist = res
            recs.append({
                "id": f"{args.name_prefix}_{split}_{len(recs)}",
                "pool": pool,
                "target": target,
                "n_steps": args.pool_size - 1,
                "prompt": render_prompt(pool, target),
                "init_state_text": render_init_state_text(pool, target),
                "answer_label": render_gold(hist, target),
                "split": split,
            })
        path = out_dir / f"{args.name_prefix}_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
