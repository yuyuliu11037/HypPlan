"""Precompute reference log-probs log π_ref(r | ctx) for DPO training pairs.

Runs the frozen SFT model (bf16, no z, no_grad) on each (ctx, r+) and (ctx, r-)
pair and sums the token log-probs over the step span. Saves as a .pt file aligned
with the pairs JSONL (same order).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def seq_logprob(model, tokenizer, ctx_text: str, tail_text: str, device) -> float:
    """Sum of token log-probs of tail_text tokens under model, conditioned on ctx_text.

    Tokenizes ctx_text and tail_text separately (so tail tokens align exactly to a
    known span). Forward on concat, extract logits at positions ctx_len-1 .. end-1
    (predicting tail tokens), and gather log-probs of the tail token ids.
    """
    ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)
    tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)
    input_ids = torch.tensor([ctx_ids + tail_ids], device=device)
    out = model(input_ids=input_ids)
    logits = out.logits[0]  # (L, V)
    # Predict token at position p from logits at p-1. We want log p(tail_t) for t in tail.
    # Tail tokens are at positions ctx_len, ctx_len+1, ..., ctx_len+tail_len-1.
    # Logits predicting them are at positions ctx_len-1, ..., ctx_len+tail_len-2.
    ctx_len = len(ctx_ids)
    tail_len = len(tail_ids)
    pred_logits = logits[ctx_len - 1 : ctx_len - 1 + tail_len]  # (tail_len, V)
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    tail_tokens = torch.tensor(tail_ids, device=device)
    selected = log_probs.gather(-1, tail_tokens.unsqueeze(-1)).squeeze(-1)  # (tail_len,)
    return selected.sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft_24_tot_merged")
    parser.add_argument("--pairs", default="data/24_train_dpo_tot.jsonl")
    parser.add_argument("--output", default="data/24_train_dpo_tot_refs.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    pairs = []
    with open(args.pairs) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Computing references for {len(pairs)} pairs")

    log_pi_ref_pos = []
    log_pi_ref_neg = []
    for i, p in enumerate(pairs):
        lp_pos = seq_logprob(model, tokenizer, p["ctx_text"], p["pos_tail"], device)
        lp_neg = seq_logprob(model, tokenizer, p["ctx_text"], p["neg_tail"], device)
        log_pi_ref_pos.append(lp_pos)
        log_pi_ref_neg.append(lp_neg)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(pairs)} | ref_pos={lp_pos:.2f} ref_neg={lp_neg:.2f}", flush=True)

    log_pi_ref_pos = torch.tensor(log_pi_ref_pos, dtype=torch.float32)
    log_pi_ref_neg = torch.tensor(log_pi_ref_neg, dtype=torch.float32)

    print(f"\nlog_pi_ref_pos: mean={log_pi_ref_pos.mean():.3f} "
          f"std={log_pi_ref_pos.std():.3f}")
    print(f"log_pi_ref_neg: mean={log_pi_ref_neg.mean():.3f} "
          f"std={log_pi_ref_neg.std():.3f}")
    margin_ref = log_pi_ref_pos - log_pi_ref_neg
    print(f"ref margin (pos - neg): mean={margin_ref.mean():.3f} "
          f"std={margin_ref.std():.3f} | >0: {(margin_ref > 0).float().mean()*100:.1f}%")

    torch.save({
        "log_pi_ref_pos": log_pi_ref_pos,
        "log_pi_ref_neg": log_pi_ref_neg,
    }, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
