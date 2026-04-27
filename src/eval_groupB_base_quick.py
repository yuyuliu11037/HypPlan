"""Quick base-accuracy check for Group B tasks.

Loads Qwen2.5-14B-Instruct in 4-bit (bitsandbytes) so the full model fits
in ~10GB and can run on any GPU. Runs greedy decoding on a small slice
of each task's test set, scores via the same `score_ood` scorers.

Goal: answer the binary question "is the base model >0% on each task?".
If 0%, that task's design is too hard and needs substitution.

4-bit numbers will be slightly worse than bf16, so this is a LOWER bound.
Non-zero in 4-bit ⇒ non-zero in bf16. Zero in 4-bit ⇒ probably zero in
bf16, but worth a confirmation later.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--out_dir", default="results/eval_groupB_base")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4-bit quantization via bitsandbytes for memory efficiency.
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs = dict(quantization_config=bnb_cfg)
        print("Loading 4-bit quantized model")
    except Exception:
        load_kwargs = dict(torch_dtype=torch.bfloat16)
        print("BitsAndBytes unavailable; loading bf16 (needs ~28GB)")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, device_map="auto",
        **load_kwargs,
    )
    model.eval()

    from src.dagger_ood_adapters import ADAPTERS
    from src.score_ood import score_rulechain, score_clutrr

    SCORERS = {
        "rulechain": score_rulechain,
        "synthlogic": score_rulechain,
        "clutrr": score_clutrr,
    }
    TEST_FILES = {
        t: f"data/{t}_test.jsonl"
        for t in ["rulechain", "synthlogic", "clutrr"]
    }

    summary = []
    for task in ["rulechain", "synthlogic", "clutrr"]:
        records = [json.loads(l) for l in open(TEST_FILES[task])][: args.limit]
        adapter_cls = ADAPTERS[task]
        out_path = out_dir / f"{task}_base_4bit.jsonl"

        print(f"\n=== {task} (n={len(records)}) ===", flush=True)
        n_correct = 0
        with open(out_path, "w") as fout:
            t0 = time.time()
            for i, rec in enumerate(records):
                # Build the same prompt the adapter would use.
                try:
                    adapter = adapter_cls(rec)
                except Exception as e:
                    fout.write(json.dumps({"id": rec.get("id"),
                                            "error": f"adapter: {e}"}) + "\n")
                    continue
                prompt_text, _ = adapter.make_prompt(tok)
                prompt_text = prompt_text + adapter.step_priming_prefix(1)
                ids = tok.encode(prompt_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        ids, max_new_tokens=args.max_new_tokens,
                        do_sample=False, temperature=1.0,
                        pad_token_id=tok.eos_token_id,
                    )
                gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
                ok, _ = SCORERS[task](gen, rec)
                if ok:
                    n_correct += 1
                fout.write(json.dumps({
                    "id": rec.get("id"), "correct": bool(ok),
                    "generation": gen[:1000],
                }) + "\n")
                fout.flush()
            elapsed = time.time() - t0
        acc = n_correct / len(records) if records else 0.0
        print(f"{task}: {n_correct}/{len(records)} = {acc:.0%} "
              f"({elapsed:.0f}s)", flush=True)
        summary.append((task, n_correct, len(records), acc))

    print("\n=== SUMMARY (4-bit quantized) ===")
    for task, c, n, a in summary:
        print(f"  {task:12s}: {c}/{n} = {a:.0%}")


if __name__ == "__main__":
    main()
