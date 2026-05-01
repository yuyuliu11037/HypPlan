"""Generate rollouts for OVM value-model training.

For each train problem, sample N trajectories from the (frozen) generator
at temperature 1.0, score each via the task's existing scorer, write a
JSONL where every record is one trajectory paired with its outcome label.

Output schema (one line per trajectory):
    {"id": str, "task": str, "rollout_idx": int,
     "prompt": str, "generation": str,
     "outcome": int (0 or 1), "scorer_meta": dict}

The generator is the same Qwen-14B-Instruct + per-task PT-SFT LoRA we
already use as a baseline. Sharded across GPUs for speed; per-rank
heartbeat logs flag any silently slow rank.

Run (per shard, on GPU r):
    CUDA_VISIBLE_DEVICES=r python3.10 -m scripts.gen_ovm_rollouts \\
        --task g24 \\
        --base_model Qwen/Qwen2.5-14B-Instruct \\
        --lora_adapter checkpoints/sft_pt_24_qwen14b/lora \\
        --train_data data/24_train.jsonl --train_limit 900 \\
        --rollouts_per_problem 40 --temperature 1.0 \\
        --max_new_tokens 256 \\
        --output data/ovm/rollouts_g24.jsonl \\
        --shard_rank r --shard_world 4
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


# Map our task names → existing scorer functions.
def _score(task: str, gen: str, rec: dict) -> int:
    """Return 1 if the generation is correct for `rec`, else 0."""
    if task == "g24":
        from src.score_ood import score_g24
        ok, _ = score_g24(gen, rec)
        return int(ok)
    if task == "nqueens":
        from src.score_ood import score_nqueens
        ok, _ = score_nqueens(gen, rec)
        return int(ok)
    if task == "bw":
        from src.score_ood import score_blocksworld_goal_reaching
        # Train data uses "question"; test data uses "prompt".
        prompt_text = rec.get("prompt") or rec.get("question") or ""
        ok, _ = score_blocksworld_goal_reaching(gen, prompt_text)
        return int(ok)
    if task == "gc":
        from src.oracle_graphcolor import (
            Problem, parse_coloring, score_coloring,
        )
        edges = tuple((int(u), int(v)) for u, v in rec["edges"])
        prob = Problem(n=int(rec["n"]), edges=edges,
                       one_solution=rec.get("gold_solution"))
        coloring = parse_coloring(gen, prob)
        return int(score_coloring(prob, coloring))
    if task == "clutrr":
        from src.score_ood import score_clutrr
        ok, _ = score_clutrr(gen, rec)
        return int(ok)
    if task == "proofwriter":
        # ProofWriter is True/False answer; parse and compare.
        import re
        m = re.search(r"Answer\s*[:\-]?\s*(True|False)", gen, re.IGNORECASE)
        if not m:
            return 0
        pred = m.group(1).lower() == "true"
        return int(pred == bool(rec.get("answer", False)))
    if task == "pq":
        import re
        m = re.search(r"Answer\s*[:\-]?\s*\b(A|B)\b", gen, re.IGNORECASE)
        if not m:
            return 0
        pred = m.group(1)
        # PQ records use 'answer_letter' or 'answer_label' to hold A/B.
        gold = rec.get("answer_letter") or rec.get("answer_label") or rec.get("answer")
        return int(pred == gold)
    if task == "rulechain":
        from src.score_ood import score_rulechain
        ok, _ = score_rulechain(gen, rec)
        return int(ok)
    raise ValueError(f"Unsupported task: {task}")


def _build_prompt(task: str, rec: dict) -> str:
    """Match the exact prompt format the PT-SFT generator was trained on
    (from src.eval_pt_ood.build_question)."""
    if task == "g24":
        # G24 PT-SFT was trained with question = the raw "Problem: ... " line
        # but the generic eval_pt_ood doesn't define g24. The 24-task uses
        # the same `Problem: x y z w` text. We inspect the train file at
        # call site; here we rely on rec["question"] if present, else build.
        if "question" in rec:
            return rec["question"]
        if "problem" in rec:
            return f"Problem: {rec['problem'].replace(',', ' ')}"
        raise ValueError("g24 record missing 'question'/'problem'")
    if task == "nqueens":
        from src.oracle_nqueens import Problem, format_question
        prob = Problem(N=int(rec["N"]),
                       prefix=tuple(rec.get("prefix", [])))
        return format_question(prob)
    if task in ("bw", "gc"):
        # PT-SFT for these tasks was trained with question = rec["question"]
        # in the *_train_sft_plan.jsonl format. Test JSONLs use 'prompt'
        # with slightly different wording — handle both.
        q = rec.get("question") or rec.get("prompt")
        if q is None:
            raise ValueError(f"{task} record missing 'question'/'prompt'")
        return q
    if task in ("clutrr", "proofwriter", "rulechain", "pq"):
        # Group B PT-SFT records use 'prompt'; train_sft_plan files use
        # 'question'. Both work as the bare task statement.
        q = rec.get("question") or rec.get("prompt")
        if q is None:
            raise ValueError(f"{task} record missing 'question'/'prompt'")
        return q
    raise ValueError(task)


def _wrap(prompt: str) -> str:
    return f"Question: {prompt}\nAnswer:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["g24", "nqueens", "bw", "gc",
                             "clutrr", "proofwriter", "pq", "rulechain"])
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--lora_adapter", required=True)
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--train_limit", type=int, default=-1)
    ap.add_argument("--rollouts_per_problem", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--heartbeat_secs", type=float, default=60.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = args.shard_rank

    # Load model.
    print(f"[r{rank}] Loading {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    print(f"[r{rank}] Attaching LoRA {args.lora_adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_adapter)
    model.eval()

    # Read records (sharded by problem id, not by trajectory).
    records = [json.loads(l) for l in open(args.train_data)]
    if args.train_limit > 0:
        records = records[: args.train_limit]
    if args.shard_world > 1:
        records = records[rank :: args.shard_world]
    print(f"[r{rank}] task={args.task} {len(records)} records, "
          f"{args.rollouts_per_problem} rollouts each, "
          f"output={args.output}", flush=True)

    # Per-shard output (avoid clobber).
    out_path = Path(args.output)
    if args.shard_world > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}_shard{rank}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last_hb = time.time()
    total_done = 0
    total_correct = 0
    n_total_traj = len(records) * args.rollouts_per_problem
    with open(out_path, "w") as fout, torch.no_grad():
        for ri, rec in enumerate(records):
            try:
                question_text = _build_prompt(args.task, rec)
            except Exception as e:
                print(f"[r{rank}] skip rec idx={ri} build_prompt error: {e}",
                      flush=True)
                continue
            prompt_text = _wrap(question_text)
            input_ids = tok.encode(prompt_text, return_tensors="pt").to(device)
            rec_id = rec.get("id", f"idx_{ri}")
            for roll in range(args.rollouts_per_problem):
                ro_t = time.time()
                try:
                    out_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tok.eos_token_id,
                    )
                    gen = tok.decode(out_ids[0, input_ids.size(1):],
                                      skip_special_tokens=False)
                except Exception as e:
                    gen = ""
                    print(f"[r{rank}] gen error rec={rec_id} roll={roll}: {e}",
                          flush=True)
                outcome = _score(args.task, gen, rec)
                ro_dt = time.time() - ro_t
                if ro_dt > 30.0:
                    print(f"  [r{rank}] SLOW rollout rec={rec_id} "
                          f"roll={roll} took {ro_dt:.1f}s", flush=True)
                fout.write(json.dumps({
                    "id": rec_id,
                    "task": args.task,
                    "rollout_idx": roll,
                    "prompt": prompt_text,
                    "generation": gen,
                    "outcome": outcome,
                }) + "\n")
                total_done += 1
                total_correct += outcome
                # Periodic heartbeat / flush.
                if time.time() - last_hb >= args.heartbeat_secs:
                    fout.flush()
                    elapsed = time.time() - t0
                    rate = total_done / max(elapsed, 1e-6)
                    eta = (n_total_traj - total_done) / max(rate, 1e-6)
                    pct_corr = total_correct / max(total_done, 1)
                    print(f"  [r{rank}] HB {total_done}/{n_total_traj} "
                          f"({elapsed/60:.1f}m, {rate:.2f}/s, "
                          f"eta={eta/60:.1f}m, p_correct={pct_corr:.2%})",
                          flush=True)
                    last_hb = time.time()
            # End-of-record summary every 25 records.
            if (ri + 1) % 25 == 0:
                fout.flush()
                print(f"  [r{rank}] {ri+1}/{len(records)} records done "
                      f"({total_done} traj, p_correct={total_correct/max(total_done,1):.2%})",
                      flush=True)
    print(f"[r{rank}] done in {(time.time()-t0)/60:.1f}m: "
          f"{total_correct}/{total_done} correct = "
          f"{total_correct/max(total_done,1):.2%}",
          flush=True)


if __name__ == "__main__":
    main()
