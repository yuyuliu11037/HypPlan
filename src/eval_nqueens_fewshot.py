"""Few-shot N-Queens eval on Qwen-14B-Instruct.

Reads a test JSONL produced by data/generate_data_nqueens.py. Each record
gives an initial state (N, k, prefix). The model is shown a 2-shot
demo prompt (N=4, N=5) and the puzzle continued from the prefix; it
must produce a Solution: line that (i) starts with the prefix and
(ii) is a valid N-Queens placement.

Sharding via --shard_rank/--shard_world: rank i handles records
i, i+world, i+2*world, ... and writes to {out_path%.jsonl}_shard{i}.jsonl.

Usage:
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python3.10 -m src.eval_nqueens_fewshot \\
      --test_data data/nqueens_n8_test.jsonl \\
      --shard_rank $i --shard_world 4 \\
      --out results/nqueens_fewshot/n8_fewshot.jsonl &
  done; wait
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                            BitsAndBytesConfig)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle_nqueens import (parse_solution, render_prefix_steps,
                                  score_solution)


# Verbatim user prompt header (2 demos). The "=== Now solve ===" section
# is appended dynamically per record.
FEWSHOT_HEADER = """Solve the N-Queens problem: place N queens on an N×N board, one per
row, so that no two queens share the same column or diagonal.

At each step, state which column you place the queen in for the
current row. After placing, list the columns that are still
available for the next row (not blocked by column or diagonal
conflicts with any placed queen).

=== Example 1 (N=4) ===

Board size: 4
Step 1: Place queen in row 1 at column 2.
  Placed: [(1,2)]
  Available for row 2: [4]  (cols 1,3 blocked by diag; col 2 blocked by col)
Step 2: Place queen in row 2 at column 4.
  Placed: [(1,2),(2,4)]
  Available for row 3: [1, 3]
Step 3: Place queen in row 3 at column 1.
  Placed: [(1,2),(2,4),(3,1)]
  Available for row 4: [3]
Step 4: Place queen in row 4 at column 3.
  Placed: [(1,2),(2,4),(3,1),(4,3)]
Solution: [2, 4, 1, 3]

=== Example 2 (N=5) ===

Board size: 5
Step 1: Place queen in row 1 at column 1.
  Placed: [(1,1)]
  Available for row 2: [3, 4, 5]
Step 2: Place queen in row 2 at column 3.
  Placed: [(1,1),(2,3)]
  Available for row 3: [5]
Step 3: Place queen in row 3 at column 5.
  Placed: [(1,1),(2,3),(3,5)]
  Available for row 4: [2]
Step 4: Place queen in row 4 at column 2.
  Placed: [(1,1),(2,3),(3,5),(4,2)]
  Available for row 5: [4]
Step 5: Place queen in row 5 at column 4.
  Placed: [(1,1),(2,3),(3,5),(4,2),(5,4)]
Solution: [1, 3, 5, 2, 4]"""


def build_prompt(tok, N: int, prefix: list[int]) -> tuple[str, int]:
    """Construct the chat-template prompt and return (text, next_row).

    `prefix` is a length-k 1-indexed column list for rows 1..k.
    The model is primed with `Step (k+1):`.
    """
    user = FEWSHOT_HEADER + "\n\n=== Now solve ===\n\nBoard size: " + str(N)
    if prefix:
        user += "\n" + render_prefix_steps(N, prefix)
    next_row = len(prefix) + 1
    msgs = [{"role": "user", "content": user}]
    chat = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    return chat + f"Step {next_row}:", next_row


def score_with_prefix(N: int, prefix: list[int],
                       generation: str) -> tuple[bool, list[int] | None]:
    sol = parse_solution(generation)
    if sol is None or len(sol) != N:
        return False, sol
    if list(sol[:len(prefix)]) != list(prefix):
        return False, sol
    return score_solution(N, sol), sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--out", default="results/nqueens_fewshot/n8_fewshot.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--use_4bit", type=int, default=1)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.shard_world > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}_shard{args.shard_rank}{out_path.suffix}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.test_data)]
    my_records = [r for i, r in enumerate(records)
                   if i % args.shard_world == args.shard_rank]
    print(f"shard {args.shard_rank}/{args.shard_world}: "
            f"{len(my_records)} records of {len(records)} total", flush=True)

    print(f"Loading {args.base_model} (4bit={bool(args.use_4bit)})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb, device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    valid = 0
    with open(out_path, "w") as f:
        for rec in my_records:
            N = rec["N"]; k = rec["k"]; prefix = rec["prefix"]
            prompt, next_row = build_prompt(tok, N, prefix)
            ids = tok.encode(prompt, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                if args.temperature <= 0:
                    out = model.generate(
                        ids, max_new_tokens=args.max_new_tokens,
                        do_sample=False, num_return_sequences=1,
                        pad_token_id=tok.eos_token_id,
                    )
                else:
                    out = model.generate(
                        ids, max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, num_return_sequences=1,
                        pad_token_id=tok.eos_token_id,
                    )
            gen_ids = out[0, ids.shape[1]:]
            gen = tok.decode(gen_ids, skip_special_tokens=True)
            full_gen = f"Step {next_row}:" + gen
            ok, sol = score_with_prefix(N, prefix, full_gen)
            valid += int(ok)
            dt = time.time() - t0
            print(f"[r{args.shard_rank}] {rec['id']} k={k} prefix={prefix} "
                    f"-> ok={ok} sol={sol} ({dt:.1f}s)", flush=True)
            f.write(json.dumps({
                "id": rec["id"], "N": N, "k": k, "prefix": prefix,
                "ok": ok, "solution": sol, "generation": full_gen,
                "latency_s": round(dt, 2),
            }) + "\n")

    print(f"\n=== shard {args.shard_rank}: {valid}/{len(my_records)} valid ===",
            flush=True)


if __name__ == "__main__":
    main()
