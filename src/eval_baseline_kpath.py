"""K-path baseline runner for ToT (top-1) and Self-Consistency (majority).

For each problem, samples K independent rollouts at temperature `temp`,
scores each with the task scorer, and aggregates:

- `--mode tot`: report `top1` (1 greedy rollout, temp=0) under the
  task adapter's structured Step-1 priming prompt. K=5 sampled
  trajectories are also generated and saved in the per-record JSONL
  for audit, but the lenient "any-of-K" oracle metric is NOT reported
  as a baseline (you'd need the gold label to pick the right one of
  K samples — not deployable).
- `--mode sc`: report `majority` (majority vote over K rollouts at
  temp>0). Canonical Self-Consistency (Wang et al. 2023).
- `--mode greedy`: just top1 (1 greedy rollout). Same as base eval but
  using the task adapter's rollout prompt.

Compatible with all 8 tasks via dagger_ood_adapters.ADAPTERS:
    pq | bw | gc | rulechain | clutrr | proofwriter | numpath | (synthlogic)
G24 has its own pipeline outside the adapter registry — use
`--task g24` and we route to a thin wrapper.

Sharding: `--shard_rank/--shard_world` for multi-GPU eval, output goes
to `{out_path%.jsonl}_shard{i}.jsonl`. Concatenate after all shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _gen(model, tok, prompt: str, max_new_tokens: int, temperature: float,
          device, num_samples: int) -> tuple[list[str], int, float]:
    """Generate `num_samples` completions from `prompt`. For temp=0, single
    deterministic sample; otherwise sample with temperature.

    Returns (decoded_texts, total_generated_tokens, latency_seconds).
    """
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-3),
        top_p=0.95 if do_sample else 1.0,
        num_return_sequences=num_samples if do_sample else 1,
        pad_token_id=tok.eos_token_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    decoded = []
    plen = ids.size(1)
    n_gen_tokens = 0
    for s in out:
        gen_ids = s[plen:]
        n_gen_tokens += int((gen_ids != tok.pad_token_id).sum().item())
        decoded.append(tok.decode(gen_ids, skip_special_tokens=True))
    if not do_sample and num_samples > 1:
        decoded = decoded * num_samples
    return decoded, n_gen_tokens, latency


# ---------------------------- Task scoring dispatch ----------------------------

def score_one(task: str, gen: str, rec: dict) -> tuple[bool, dict]:
    if task == "pq":
        from src.score_ood import score_prontoqa
        ok = score_prontoqa(gen, rec["answer_label"])
        return ok, {}
    if task == "bw":
        from src.score_ood import score_blocksworld_goal_reaching
        return score_blocksworld_goal_reaching(gen, rec["prompt"])
    if task == "gc":
        from src.oracle_graphcolor import (
            Problem, parse_coloring, score_coloring,
        )
        p = Problem(n=rec["n"], edges=tuple(map(tuple, rec["edges"])))
        coloring = parse_coloring(gen, p)
        return score_coloring(p, coloring), {}
    if task in ("rulechain", "synthlogic"):
        from src.score_ood import score_rulechain
        return score_rulechain(gen, rec)
    if task == "clutrr":
        from src.score_ood import score_clutrr
        return score_clutrr(gen, rec)
    if task == "proofwriter":
        from src.score_ood import score_proofwriter
        return score_proofwriter(gen, rec)
    if task == "numpath":
        from src.score_ood import score_numpath
        return score_numpath(gen, rec)
    if task == "g24":
        # 24-Game records use the existing varied schema; check via simple
        # arithmetic simulation.
        from src.score_ood import score_g24 as score
        return score(gen, rec)
    if task == "nqueens":
        from src.score_ood import score_nqueens
        return score_nqueens(gen, rec)
    raise ValueError(task)


def extract_answer_key(task: str, gen: str, rec: dict):
    """Return a hashable "final answer" key from the generation, used for
    majority vote in SC mode. None if unparseable."""
    if task == "pq":
        import re
        m = re.search(r"\b(A|B)\b", gen.split("\n", 1)[0])
        if m:
            return m.group(1)
        return None
    if task in ("rulechain", "synthlogic", "proofwriter"):
        # All three end with `Answer: <X>`; capture `<X>`.
        import re
        m = re.search(r"Answer\s*[:\-]?\s*(.+?)\s*$", gen, re.MULTILINE)
        return m.group(1).strip() if m else None
    if task == "clutrr":
        from src.oracle_clutrr import parse_answer
        return parse_answer(gen)
    if task == "numpath":
        # Use the FINAL achieved value as the answer key.
        ok, info = score_one(task, gen, rec)
        return info.get("final")
    if task == "bw":
        # Use the canonical-form action sequence.
        from src.score_ood import extract_blocksworld_plan
        return tuple(extract_blocksworld_plan(gen))
    if task == "gc":
        from src.oracle_graphcolor import Problem, parse_coloring
        p = Problem(n=rec["n"], edges=tuple(map(tuple, rec["edges"])))
        col = parse_coloring(gen, p)
        return tuple(sorted(col.items())) if col else None
    if task == "g24":
        import re
        m = re.search(r"Answer\s*[:\-]?\s*(-?\d+)", gen)
        return int(m.group(1)) if m else None
    if task == "nqueens":
        from src.oracle_nqueens import parse_solution
        sol = parse_solution(gen)
        return tuple(sol) if sol else None
    return None


# ---------------------------- Prompt building ----------------------------

def build_prompt(task: str, rec: dict, tok) -> str:
    """For all tasks except g24, route through the dagger_ood_adapter's
    make_prompt + step priming. For g24 we use the existing v1 prompt
    builder."""
    if task == "g24":
        # Use a CoT-style prompt for G24. Two schemas:
        # - 24_test.jsonl: {"problem": "a,b,c,d", ...} target=24
        # - 24_varied_bal_test.jsonl: {"pool": [...], "target": int, ...}
        if "pool" in rec:
            pool = " ".join(str(n) for n in rec["pool"])
            target = rec["target"]
        else:
            pool = " ".join(rec["problem"].split(","))
            target = int(rec.get("target", 24))
        sys_msg = (
            "You will solve a Game-of-24-like puzzle. Use each number "
            "exactly once and the four operations (+, -, *, /) to make "
            "the target. Output one operation per step.\n"
            "Output format:\n"
            "  Step 1: a op b = r\n"
            "  Step 2: c op d = s\n"
            "  ...\n"
            "  Answer: <target>"
        )
        user = f"Numbers: {pool}\nTarget: {target}"
        msgs = [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    from src.dagger_ood_adapters import ADAPTERS
    adapter = ADAPTERS[task](rec)
    prompt, _ = adapter.make_prompt(tok)
    return prompt + adapter.step_priming_prefix(1)


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                     choices=["pq", "bw", "gc", "rulechain", "clutrr",
                              "proofwriter", "numpath", "g24", "synthlogic",
                              "nqueens"])
    ap.add_argument("--mode", required=True,
                     choices=["greedy", "tot", "sc"])
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--use_4bit", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    if args.shard_world > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}_shard{args.shard_rank}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[r{args.shard_rank}] Loading {args.base_model} "
          f"(4bit={bool(args.use_4bit)})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    is_mistral3 = "mistral-small-3" in args.base_model.lower()
    is_gpt_oss = "gpt-oss" in args.base_model.lower()
    if is_mistral3:
        from transformers.models.mistral3 import (
            Mistral3ForConditionalGeneration,
        )
        loader = Mistral3ForConditionalGeneration
    else:
        loader = AutoModelForCausalLM
    # GPT-OSS ships pre-quantized with mxfp4 — skip bnb to avoid double-quant.
    use_bnb = bool(args.use_4bit) and not is_gpt_oss
    if use_bnb:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = loader.from_pretrained(
            args.base_model, trust_remote_code=True, device_map="auto",
            quantization_config=bnb_cfg,
        )
    elif is_gpt_oss:
        model = loader.from_pretrained(
            args.base_model, trust_remote_code=True, device_map="auto",
        )
    else:
        model = loader.from_pretrained(
            args.base_model, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
    model.eval()

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"[r{args.shard_rank}] mode={args.mode} task={args.task} "
          f"records={len(records)} K={args.K}", flush=True)

    n_top1 = 0; n_any = 0; n_majority = 0
    t0 = time.time()
    with open(out_path, "w") as fout, torch.no_grad():
        for i, rec in enumerate(records):
            prompt = build_prompt(args.task, rec, tok)
            if is_gpt_oss:
                prompt = prompt + "<|channel|>final<|message|>"

            top1_gen = ""
            top1_ok = False
            top1_tokens = 0
            top1_latency = 0.0
            if args.mode in ("greedy", "tot"):
                top1_gens, top1_tokens, top1_latency = _gen(
                    model, tok, prompt, args.max_new_tokens, 0.0, device, 1,
                )
                top1_gen = top1_gens[0]
                top1_ok, _ = score_one(args.task, top1_gen, rec)
                if top1_ok:
                    n_top1 += 1

            sample_gens: list[str] = []
            sample_oks: list[bool] = []
            sc_tokens = 0
            sc_latency = 0.0
            if args.mode in ("tot", "sc"):
                sample_gens, sc_tokens, sc_latency = _gen(
                    model, tok, prompt, args.max_new_tokens, args.temperature,
                    device, args.K,
                )
                sample_oks = []
                for g in sample_gens:
                    ok, _ = score_one(args.task, g, rec)
                    sample_oks.append(ok)
                if any(sample_oks):
                    n_any += 1

            majority_ok = False
            if args.mode == "sc":
                keys = [extract_answer_key(args.task, g, rec)
                         for g in sample_gens]
                keys_str = [str(k) if k is not None else None for k in keys]
                cnt = Counter(k for k in keys_str if k is not None)
                if cnt:
                    top_key, _ = cnt.most_common(1)[0]
                    for g, ok, k in zip(sample_gens, sample_oks, keys_str):
                        if k == top_key and ok:
                            majority_ok = True
                            break
                if majority_ok:
                    n_majority += 1

            fout.write(json.dumps({
                "id": rec.get("id"),
                "task": args.task, "mode": args.mode,
                "top1_ok": bool(top1_ok),
                "any_ok": bool(any(sample_oks)) if sample_oks else None,
                "majority_ok": bool(majority_ok) if args.mode == "sc" else None,
                "n_samples": len(sample_gens),
                "top1_gen": top1_gen[:600] if top1_gen else None,
                "sample_gens": [g[:600] for g in sample_gens],
                "n_gen_tokens": int(top1_tokens + sc_tokens),
                "latency_s": float(top1_latency + sc_latency),
                "sample_oks": sample_oks,
            }) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                msg = f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                if args.mode in ("greedy", "tot"):
                    msg += f"top1={n_top1/(i+1):.0%} "
                if args.mode == "sc":
                    msg += f"maj={n_majority/(i+1):.0%} "
                msg += f"rate={rate:.2f}/s"
                print(msg, flush=True)

    elapsed = time.time() - t0
    n = len(records)
    print(f"\n[r{args.shard_rank}] {args.task} {args.mode} (n={n}, K={args.K}) "
          f"({elapsed:.0f}s):")
    if args.mode in ("greedy", "tot"):
        print(f"  top1: {n_top1}/{n} = {n_top1/n:.0%}")
    if args.mode == "sc":
        print(f"  maj:  {n_majority}/{n} = {n_majority/n:.0%}")


if __name__ == "__main__":
    main()
