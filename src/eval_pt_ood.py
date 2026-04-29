"""Eval Qwen2.5-14B + planning-tokens LoRA on the 3 OOD tasks.

For each task, builds a "Question: <prompt>\\nAnswer:" prompt that matches
the SFT training format from data/prepare_pt_ood_data.py, generates greedy,
and writes raw output for downstream scoring.

Tasks:
- cd:  prompt = "<pool> | Target: <target>"; expects ".../Answer: N" output
- bw:  prompt = full PlanBench query
- pq:  prompt = init_state_text (rendered initial state); expects "Answer: A/B"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_question(task: str, rec: dict) -> str:
    if task == "cd":
        pool = " ".join(str(n) for n in rec["pool"])
        return f"{pool} | Target: {rec['target']}"
    if task == "bw":
        return rec["prompt"]
    if task == "pq":
        return rec["init_state_text"]
    if task == "gc":
        # SFT was trained with question = format_question(problem)
        return rec["init_state_text"]
    if task in ("rulechain", "synthlogic", "clutrr", "lineq",
                 "proofwriter", "numpath"):
        # Group B: SFT-PT was trained with question = rec["prompt"]
        # (matches data/annotate_sft_plan_groupB.py).
        return rec["prompt"]
    if task == "nqueens":
        # N-Queens test records carry only N/k/prefix/gold_extension.
        # Reconstruct the same question text the PT-SFT annotator used.
        from src.oracle_nqueens import Problem, format_question
        prob = Problem(N=int(rec["N"]),
                       prefix=tuple(rec.get("prefix", [])))
        return format_question(prob)
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                     choices=["cd", "bw", "pq", "gc",
                              "rulechain", "synthlogic", "clutrr",
                              "lineq", "proofwriter", "numpath",
                              "nqueens"])
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--lora_adapter", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    is_gpt_oss = "gpt-oss" in args.base_model.lower()
    if is_gpt_oss:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True, device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
    print(f"Attaching LoRA {args.lora_adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_adapter)
    model.eval()

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval task={args.task} on {len(records)} records "
          f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            q = build_question(args.task, rec)
            prompt = f"Question: {q}\nAnswer:"
            input_ids = tok.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out_ids[0, input_ids.size(1):],
                              skip_special_tokens=False)
            fout.write(json.dumps({**rec, "question": q,
                                     "generation": gen}) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                       f"({rate:.2f}/s)", flush=True)
    print(f"  [r{args.shard_rank}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
