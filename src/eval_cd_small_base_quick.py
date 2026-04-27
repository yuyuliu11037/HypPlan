"""Quick 4-bit base-accuracy check for small-scale Countdown.

cd_small doesn't go through the OOD adapter pipeline yet (no adapter
needed for base eval — just prompt + scorer). This is a standalone
runner mirroring `src/eval_groupB_base_quick.py` but using the
`prompt` field directly and `score_cd_small` from `src.score_ood`.

Goal: answer "is base Qwen-14B in the 30-70% headroom range on
small-scale Countdown?" Currently lineq is 99% so we're considering
cd_small as the Group A OOD #1 replacement.
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
    ap.add_argument("--test_data", default="data/cd_small_test.jsonl")
    ap.add_argument("--out_path",
                     default="results/eval_groupB_base/cd_small_base_4bit.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    args = ap.parse_args()

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Loading 4-bit quantized model")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, device_map="auto",
        quantization_config=bnb_cfg,
    )
    model.eval()

    from src.score_ood import score_cd_small

    sys_msg = (
        "You will solve a small Countdown puzzle. Use each number exactly "
        "once and the four arithmetic operations (+, -, *, /) to make the "
        "target. Output one operation per step.\n"
        "Output format:\n"
        "  Step 1: a op b = r\n"
        "  Step 2: c op d = s\n"
        "  ...\n"
        "  Answer: <target>\n"
        "Subtraction must give a non-negative result. Division must be exact."
    )

    records = [json.loads(l) for l in open(args.test_data)][: args.limit]
    n_correct = 0
    t0 = time.time()
    with open(args.out_path, "w") as fout:
        for i, rec in enumerate(records):
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": rec["prompt"]},
            ]
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tok.encode(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
            ok, info = score_cd_small(gen, rec)
            if ok:
                n_correct += 1
            fout.write(json.dumps({
                "id": rec["id"], "correct": bool(ok),
                "generation": gen[:1000], **info,
            }) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(records)} acc={n_correct/(i+1):.0%} "
                      f"rate={rate:.2f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"\ncd_small (4-bit): {n_correct}/{len(records)} = "
          f"{n_correct/len(records):.0%} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
