"""Eval: trained LoRA + head + up_projector on varied-target (pool, target).

Reuses rollout_one from dagger_rollout_varied in greedy eval mode. Works for
both in-domain G24-varied test and OOD Countdown test (both share the same
{pool, target} schema).
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

from src.head import HyperbolicHead, UpProjector
from src.dagger_rollout_varied import rollout_one
from src.prompt_builders import (
    fewshot_chat_prompt_24, fewshot_chat_prompt_cd,
)


def _wrap_24(tokenizer, pool, target):
    problem = ",".join(str(int(x)) for x in pool)
    return fewshot_chat_prompt_24(tokenizer, problem)


def _wrap_cd(tokenizer, pool, target):
    return fewshot_chat_prompt_cd(tokenizer, pool, int(target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="e.g. checkpoints/dagger_stage2_24_varied_qwen14b/z_s1234")
    ap.add_argument("--test_data", required=True,
                    help="jsonl with {pool, target} per line")
    ap.add_argument("--output", required=True)
    ap.add_argument("--use_z", action="store_true")
    ap.add_argument("--random_z", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", choices=["generic", "task24", "cd"],
                    default="generic",
                    help="Which fewshot prompt format to use at eval time. "
                         "'task24'=fixed-24, 'cd'=Countdown, 'generic'=trained format.")
    ap.add_argument("--shard_rank", type=int, default=0,
                    help="0-indexed shard id (for parallel multi-GPU eval).")
    ap.add_argument("--shard_world", type=int, default=1,
                    help="Number of shards (one process per shard).")
    ap.add_argument("--head_override", type=str, default=None,
                    help="Override the head checkpoint path (e.g. to use a "
                         "task-specific head at eval time).")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    with open(ckpt_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = cfg["model"]["base_model"]
    head_path = args.head_override or cfg["model"]["head_checkpoint"]

    print(f"Loading base {base_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
    ).to(device)

    print(f"Loading LoRA {ckpt_dir}/lora", flush=True)
    model = PeftModel.from_pretrained(base, str(ckpt_dir / "lora"))
    model.eval()

    print(f"Loading head {head_path}", flush=True)
    sd = torch.load(head_path, map_location=device, weights_only=False)
    in_dim = sd["in_dim"]
    mc = sd["config"]["model"]
    head = HyperbolicHead(in_dim=in_dim, hyp_dim=mc["hyp_dim"],
                           hidden_dims=mc["head_hidden_dims"],
                           manifold=mc["manifold"]).to(device).float()
    head.load_state_dict(sd["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    up_in = mc["hyp_dim"] + (1 if mc["manifold"] == "lorentz" else 0)
    up_proj = UpProjector(in_dim=up_in,
                           hidden=int(cfg["model"]["up_proj_hidden"]),
                           out_dim=base.config.hidden_size).to(device).float()
    up_proj.load_state_dict(
        torch.load(ckpt_dir / "up_projector.pt", map_location=device,
                   weights_only=False))
    up_proj.eval()

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Evaluating on {len(records)} records (use_z={args.use_z} "
          f"random_z={args.random_z})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_solved = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            pool = rec["pool"]; target = rec["target"]
            pb = {"generic": None, "task24": _wrap_24, "cd": _wrap_cd}[args.prompt]
            r = rollout_one(model, tokenizer, head, up_proj, pool, target,
                             device, use_z=args.use_z,
                             temperature=args.temperature, top_p=1.0,
                             max_new_tokens=args.max_new_tokens,
                             max_steps=args.max_steps, random_z=args.random_z,
                             prompt_builder=pb)
            n_solved += int(r.solved)
            fout.write(json.dumps({
                "pool": pool, "target": target,
                "solved": r.solved,
                "stopped_reason": r.stopped_reason,
                "generation": r.generation_text,
            }) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(records)} solved={n_solved}={n_solved/(i+1):.2%} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"FINAL: {n_solved}/{len(records)} = {n_solved/len(records):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
