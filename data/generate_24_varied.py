"""Generate varied-target Game-of-24 datasets from existing G24 trees.

For each unique problem in `data/24_{train,val,test}.jsonl`, enumerates the
tree and extracts every valid (pool, target) pair reachable through integer
states. Subsamples up to `--pairs_per_problem` pairs per source problem,
stratified by n_steps (1, 2, 3) so each depth is represented.

Also injects trivial 0-step pairs ([k], k) for k in 1..24 so the model sees
the "pool already equals target" case.

Global dedup across splits by (sorted pool, target): if a pair appears in
train's dedup set, it won't be re-emitted in val or test, preventing data
leakage.

Output JSONL schema per record:
  {"pool": [int, ...], "target": int, "n_steps": int,
   "source_problem": "a,b,c,d", "split": "train|val|test"}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.varied_24 import collect_unique_pairs


def unique_problems(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with path.open() as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def sample_per_problem(pairs: list[dict], k: int, rng: random.Random
                       ) -> list[dict]:
    """Stratified sample: split pairs by n_steps, take roughly k/B from each."""
    by_steps: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_steps[p["n_steps"]].append(p)
    if not by_steps:
        return []
    steps = sorted(by_steps.keys())
    per_bucket = max(1, k // len(steps))
    out: list[dict] = []
    for s in steps:
        bucket = by_steps[s]
        rng.shuffle(bucket)
        out.extend(bucket[:per_bucket])
    # Top up if we're under k (e.g., some bucket was smaller than per_bucket).
    remaining = k - len(out)
    if remaining > 0:
        leftovers: list[dict] = []
        for s in steps:
            leftovers.extend(by_steps[s][per_bucket:])
        rng.shuffle(leftovers)
        out.extend(leftovers[:remaining])
    return out


def trivial_zero_step_pairs() -> list[dict]:
    """Inject [k]/target=k for small k so 0-step trivial cases appear."""
    return [
        {"pool": [k], "target": k, "n_steps": 0,
         "source_problem": f"trivial_{k}"}
        for k in range(1, 25)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in",  default="data/24_train.jsonl")
    ap.add_argument("--val_in",    default="data/24_val.jsonl")
    ap.add_argument("--test_in",   default="data/24_test.jsonl")
    ap.add_argument("--train_out", default="data/24_varied_train.jsonl")
    ap.add_argument("--val_out",   default="data/24_varied_val.jsonl")
    ap.add_argument("--test_out",  default="data/24_varied_test.jsonl")
    ap.add_argument("--pairs_per_problem", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_keys: set = set()  # global dedup across splits

    splits = [
        ("train", Path(args.train_in), Path(args.train_out), True),
        ("val",   Path(args.val_in),   Path(args.val_out),   False),
        ("test",  Path(args.test_in),  Path(args.test_out),  False),
    ]

    for split_name, in_path, out_path, inject_trivial in splits:
        problems = unique_problems(in_path)
        written = 0
        depth_counts: dict[int, int] = defaultdict(int)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w") as fout:
            # Trivial 0-step pairs only in training to teach the "already
            # equals" case — we don't evaluate on them.
            if inject_trivial:
                for rec in trivial_zero_step_pairs():
                    key = (tuple(rec["pool"]), rec["target"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    rec["split"] = split_name
                    fout.write(json.dumps(rec) + "\n")
                    written += 1
                    depth_counts[rec["n_steps"]] += 1

            for i, problem in enumerate(problems):
                all_pairs = collect_unique_pairs(problem)
                sampled = sample_per_problem(all_pairs, args.pairs_per_problem,
                                             rng)
                for rec in sampled:
                    key = (tuple(rec["pool"]), rec["target"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    rec["split"] = split_name
                    fout.write(json.dumps(rec) + "\n")
                    written += 1
                    depth_counts[rec["n_steps"]] += 1
                if (i + 1) % 200 == 0:
                    print(f"  {split_name}: {i+1}/{len(problems)} problems, "
                          f"{written} pairs so far", flush=True)

        depth_str = ", ".join(f"d{d}={c}" for d, c in
                              sorted(depth_counts.items()))
        print(f"{split_name}: wrote {written} pairs ({depth_str}) → {out_path}",
              flush=True)


if __name__ == "__main__":
    main()
