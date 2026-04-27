"""Generate synthlogic OOD test data.

Synthlogic = rule-chaining at OOD difficulty. Same Horn-clause primitive
as `rulechain`, but harder generator parameters and a different predicate
vocabulary so we don't leak surface tokens between training and eval.

OOD eval defaults: depth in {5,6,7}, 24 predicates, 30 rules,
`pred_prefix='q'` (training uses 'p').
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_data_rulechain import gen_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=2000,
                     help="For in-domain Stage-2/SFT-PT training; can be 0")
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--depths", default="5,6,7")
    ap.add_argument("--n_predicates", type=int, default=24)
    ap.add_argument("--n_rules", type=int, default=30)
    ap.add_argument("--seed_base", type=int, default=5678)
    ap.add_argument("--name_prefix", default="synthlogic")
    ap.add_argument("--pred_prefix", default="q")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
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
        print(f"Generating {args.name_prefix}/{split}: n={n}, depths={depths}, "
              f"n_pred={args.n_predicates}, n_rules={args.n_rules}, "
              f"prefix='{args.pred_prefix}'", flush=True)
        recs = gen_split(
            n=n, depths=depths,
            n_predicates=args.n_predicates, n_rules=args.n_rules,
            seed_offset=seed, split=split, task_name=args.name_prefix,
            pred_prefix=args.pred_prefix,
        )
        path = out_dir / f"{args.name_prefix}_{split}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} records -> {path}", flush=True)


if __name__ == "__main__":
    main()
