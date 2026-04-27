"""Eval Stage-2 in-domain LoRA + per-task Stage-1 head on PQ / BW / GC.

Loads:
  - Base Qwen-2.5-14B-Instruct
  - Stage-2 LoRA (the in-domain checkpoint)
  - Stage-2 UpProjector (from the same checkpoint dir)
  - Stage-1 head (per-task)

For each test record: build adapter, run greedy rollout with z-injection at
each step boundary (use_z=True, temperature=0). Score "solved" via the
adapter (PQ: derived answer letter matches gold; BW: goal reached; GC: all
vertices colored without conflicts).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dagger_ood_adapters import ADAPTERS
from src.dagger_rollout_ood import rollout_one
from src.head import HyperbolicHead, UpProjector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--ckpt_dir", required=True,
                     help="Stage-2 in-domain checkpoint dir, e.g. "
                          "checkpoints/dagger_stage2_pq_indomain")
    ap.add_argument("--head_path", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--use_z", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=14)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"Attaching LoRA {args.ckpt_dir}/lora", flush=True)
    model = PeftModel.from_pretrained(base, str(Path(args.ckpt_dir) / "lora"))
    model.eval()

    sd = torch.load(args.head_path, map_location=device, weights_only=False)
    in_dim = sd["in_dim"]
    mc = sd["config"]["model"]
    head = HyperbolicHead(in_dim=in_dim, hyp_dim=mc["hyp_dim"],
                           hidden_dims=mc["head_hidden_dims"],
                           manifold=mc["manifold"]).to(device).float()
    head.load_state_dict(sd["state_dict"]); head.eval()
    for p in head.parameters(): p.requires_grad = False

    with open(Path(args.ckpt_dir) / "config.yaml") as f:
        ckpt_cfg = yaml.safe_load(f)
    up_in = mc["hyp_dim"] + (1 if mc["manifold"] == "lorentz" else 0)
    up_proj = UpProjector(in_dim=up_in,
                            hidden=int(ckpt_cfg["model"]["up_proj_hidden"]),
                            out_dim=base.config.hidden_size).to(device).float()
    up_proj.load_state_dict(torch.load(
        Path(args.ckpt_dir) / "up_projector.pt", map_location=device,
        weights_only=False))
    up_proj.eval()

    AdapterCls = ADAPTERS[args.task]
    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval Stage-2 in-domain {args.task} on {len(records)} records "
            f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_correct = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            try:
                adapter = AdapterCls(rec)
            except Exception as e:
                fout.write(json.dumps({**{k: rec[k] for k in rec
                                            if k != "prompt"},
                                         "error": f"adapter: {e}"[:200]}
                                        ) + "\n")
                fout.flush()
                continue
            try:
                r = rollout_one(model, tok, head, up_proj, adapter, device,
                                  use_z=bool(args.use_z), temperature=0.0,
                                  top_p=1.0,
                                  max_new_tokens=args.max_new_tokens,
                                  max_steps=args.max_steps,
                                  random_z=False)
            except Exception as e:
                fout.write(json.dumps({**{k: rec[k] for k in rec
                                            if k != "prompt"},
                                         "error": f"rollout: {type(e).__name__} {e}"[:200]
                                        }) + "\n")
                fout.flush()
                continue

            # Per-task correctness check
            ok = False
            extra = {}
            if args.task == "pq":
                # Derived answer letter must match gold_label
                gold = rec.get("answer_label")
                derived = adapter._answer_letter(r.final_state) \
                    if r.solved else None
                ok = (derived is not None and gold is not None
                        and derived == gold)
                extra["derived"] = derived
            elif args.task == "bw":
                ok = bool(r.solved)
            elif args.task == "gc":
                ok = bool(r.solved)
            elif args.task in ("rulechain", "synthlogic", "clutrr"):
                # All Group B tasks: solved iff adapter.is_solved on
                # final state. rollout_one already checks this.
                ok = bool(r.solved)
            if ok:
                n_correct += 1
            fout.write(json.dumps({
                **{k: rec[k] for k in rec if k != "prompt"},
                "solved": bool(r.solved),
                "stopped_reason": r.stopped_reason,
                "generation": r.generation_text,
                "n_boundaries": len(r.boundaries),
                "correct": ok,
                **extra,
            }) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or i == len(records) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(records) - (i + 1)) / rate
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                        f"acc={n_correct/(i+1):.3f} "
                        f"({elapsed/60:.1f}m, eta={eta/60:.1f}m)",
                        flush=True)
    print(f"  [r{args.shard_rank}] done. correct {n_correct}/{len(records)}",
            flush=True)


if __name__ == "__main__":
    main()
