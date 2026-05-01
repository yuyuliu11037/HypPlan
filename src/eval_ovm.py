"""OVM step-level value-guided beam search.

For each test record:
  1. Build the same `Question: <q>\nAnswer:` prompt format the PT-SFT
     generator was trained with.
  2. Maintain a beam of partial trajectories. Initially: one beam with
     empty trajectory.
  3. At each step iteration (up to max_steps):
       a. For each non-terminal beam, sample K continuations from the
          generator, each stopping at the next newline (or EOS).
       b. Score each candidate by feeding the full prompt+candidate
          through the model + value head; take the value of the last
          token.
       c. Keep the top-b candidates by value.
  4. Return the highest-final-value beam's full generation, parse with
     the task's existing scorer.

This is a straightforward implementation — no batched-candidate
optimization yet. ~2-3 hours per 200-record task at K=20 b=5; we can
optimize after the G24 sanity check if needed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ovm_head import ValueHead


_NL_TOKEN_ID_CACHE: dict = {}


def _newline_token_ids(tok) -> list[int]:
    """Token ids that contain a literal newline character. Used as
    stopping criterion for step-level decoding."""
    cached = _NL_TOKEN_ID_CACHE.get(id(tok))
    if cached is not None:
        return cached
    out = []
    vocab = tok.get_vocab()
    for s, tid in vocab.items():
        if "\n" in s:
            out.append(tid)
    _NL_TOKEN_ID_CACHE[id(tok)] = out
    return out


def _build_prompt(task: str, rec: dict) -> str:
    if task == "g24":
        if "question" in rec:
            q = rec["question"]
        elif "problem" in rec:
            q = f"Problem: {rec['problem'].replace(',', ' ')}"
        else:
            raise ValueError("g24 record missing 'question'/'problem'")
        return f"Question: {q}\nAnswer:"
    if task == "nqueens":
        from src.oracle_nqueens import Problem, format_question
        prob = Problem(N=int(rec["N"]),
                       prefix=tuple(rec.get("prefix", [])))
        return f"Question: {format_question(prob)}\nAnswer:"
    if task in ("bw", "gc"):
        q = rec.get("question") or rec.get("prompt")
        if q is None:
            raise ValueError(f"{task} record missing 'question'/'prompt'")
        return f"Question: {q}\nAnswer:"
    if task in ("clutrr", "proofwriter", "rulechain", "pq"):
        q = rec.get("question") or rec.get("prompt")
        if q is None:
            raise ValueError(f"{task} record missing 'question'/'prompt'")
        return f"Question: {q}\nAnswer:"
    raise ValueError(task)


def _terminal(text: str) -> bool:
    """Has the trajectory emitted a final answer marker?"""
    if re.search(r"^\s*Answer\s*:", text, re.MULTILINE):
        return True
    if re.search(r"^\s*Solution\s*:", text, re.MULTILINE):
        return True
    if "<PLAN:ANS>" in text:
        return True
    return False


@torch.no_grad()
def _sample_step(model, tok, prompt, current_text, device,
                 max_step_tokens, temperature, top_p, K):
    """Sample K continuations of `prompt + current_text`, each up to
    `max_step_tokens` or the next newline. Returns list of K text strings
    (just the new step content)."""
    full_text = prompt + current_text
    full_ids = tok.encode(full_text, add_special_tokens=False,
                          return_tensors="pt").to(device)
    bs_input = full_ids.repeat(K, 1)
    out = model.generate(
        bs_input,
        max_new_tokens=max_step_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
    )
    new = out[:, full_ids.size(1):]
    step_texts = []
    for i in range(K):
        gen = tok.decode(new[i], skip_special_tokens=False)
        # Truncate at first newline so each "step" is one line.
        nl = gen.find("\n")
        if nl >= 0:
            step_texts.append(gen[: nl + 1])   # include the newline
        else:
            step_texts.append(gen)
    return step_texts


@torch.no_grad()
def _score_value(model, head, tok, full_text, device):
    """Single-text version (kept for the initial empty-beam scoring)."""
    ids = tok.encode(full_text, add_special_tokens=False,
                      return_tensors="pt").to(device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1][:, -1, :]
    return head(h.float()).item()


@torch.no_grad()
def _score_value_batch(model, head, tok, texts, device):
    """Batched scoring of the last token of each text. Pads on the LEFT
    so the last position is always `seq_len-1`. Returns a list of floats
    in (0, 1) of len(texts)."""
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False)
    # Pad on the left so the final token is at position -1 for all rows.
    # Hugging Face default is right-padding; do it manually.
    ids = enc["input_ids"]
    am = enc["attention_mask"]
    B, T = ids.shape
    new_ids = torch.full_like(ids, tok.pad_token_id)
    new_am = torch.zeros_like(am)
    for i in range(B):
        n = int(am[i].sum().item())
        new_ids[i, T - n:] = ids[i, :n]
        new_am[i, T - n:] = 1
    new_ids = new_ids.to(device)
    new_am = new_am.to(device)
    out = model(input_ids=new_ids, attention_mask=new_am,
                output_hidden_states=True, use_cache=False)
    h_last = out.hidden_states[-1][:, -1, :]   # (B, H)
    return head(h_last.float()).cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["g24", "nqueens", "bw", "gc",
                             "clutrr", "proofwriter", "pq", "rulechain"])
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--lora_adapter", required=True,
                    help="generator LoRA (PT-SFT)")
    ap.add_argument("--value_head", required=True,
                    help="path to ovm_head.pt from train_ovm.py")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--K", type=int, default=20, help="candidates per step")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--max_step_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--heartbeat_secs", type=float, default=60.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = args.shard_rank
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

    # Load value head.
    sd = torch.load(args.value_head, map_location=device, weights_only=False)
    hidden_dim = int(sd["hidden_dim"])
    head = ValueHead(hidden_dim).to(device).float()
    head.load_state_dict(sd["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[rank :: args.shard_world]
    print(f"[r{rank}] OVM eval task={args.task} {len(records)} records "
          f"K={args.K} beam={args.beam} max_steps={args.max_steps}",
          flush=True)

    out_path = Path(args.output)
    if args.shard_world > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}_shard{rank}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last_hb = time.time()
    n_done = 0
    with open(out_path, "w") as fout:
        for ri, rec in enumerate(records):
            try:
                prompt = _build_prompt(args.task, rec)
            except Exception as e:
                print(f"[r{rank}] skip rec idx={ri} prompt error: {e}",
                      flush=True)
                continue

            # Initial beam: empty trajectory with score of empty-prompt's value.
            beams = [{"text": "", "score": _score_value(model, head, tok,
                                                          prompt, device),
                      "ended": False}]

            for step_n in range(1, args.max_steps + 1):
                expanded = []
                # 1. Sample K continuations per non-terminal beam, collect.
                pending = []   # list of (text_after, ended)
                for bm in beams:
                    if bm["ended"]:
                        expanded.append(bm)
                        continue
                    cands = _sample_step(
                        model, tok, prompt, bm["text"], device,
                        args.max_step_tokens, args.temperature, args.top_p,
                        args.K,
                    )
                    for ct in cands:
                        new_text = bm["text"] + ct
                        ended = _terminal(new_text) or ct == ""
                        pending.append((new_text, ended))
                # 2. Batch-score everything pending.
                if pending:
                    full_texts = [prompt + t for t, _ in pending]
                    vs = _score_value_batch(model, head, tok, full_texts,
                                              device)
                    for (t, ended), v in zip(pending, vs):
                        expanded.append({"text": t, "score": v,
                                         "ended": ended})
                # Top-b by score.
                expanded.sort(key=lambda x: -x["score"])
                beams = expanded[: args.beam]
                if all(b["ended"] for b in beams):
                    break

            # Pick best beam.
            beams.sort(key=lambda x: -x["score"])
            best = beams[0]
            fout.write(json.dumps({
                **rec,   # keep all fields including "prompt" for downstream scoring
                "ovm_generation": best["text"],
                "ovm_value": best["score"],
                "ovm_n_steps": best["text"].count("\n"),
            }) + "\n")
            fout.flush()
            n_done += 1
            now = time.time()
            if now - last_hb >= args.heartbeat_secs:
                rate = n_done / max(now - t0, 1e-6)
                eta = (len(records) - n_done) / max(rate, 1e-6)
                print(f"[r{rank}] HB {n_done}/{len(records)} "
                      f"({(now-t0)/60:.1f}m, {rate:.3f}/s, "
                      f"eta={eta/60:.1f}m)",
                      flush=True)
                last_hb = now

    print(f"[r{rank}] done in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
