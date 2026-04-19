"""Inference for CoT-SFT model on Countdown."""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.dataset_cd import make_prompt


def load_model(base_model: str, adapter_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, pool: list[int], target: int,
             max_new_tokens: int = 384, temperature: float = 0.0,
             device: str = "cuda") -> str:
    prompt = make_prompt(pool, target)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if temperature <= 0:
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
    else:
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                    do_sample=True, temperature=temperature,
                                    pad_token_id=tokenizer.pad_token_id)

    generated = output_ids[0, input_ids.size(1):]
    return "Step 1:" + tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test_data", default="data/cd_test_sft.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records = []
    with open(args.test_data) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} problems from {args.test_data}")

    model, tokenizer = load_model(args.base_model, args.adapter)
    device = next(model.parameters()).device

    with open(args.output, "w") as fout:
        for i, record in enumerate(records):
            gen = generate(model, tokenizer, record["pool"], record["target"],
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, device=device)
            fout.write(json.dumps({
                "pool": record["pool"], "target": record["target"],
                "problem_idx": record["problem_idx"],
                "ground_truth": record["text"],
                "generation": gen,
            }) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0:
                print(f"Generated {i+1}/{len(records)}")
    print(f"Saved {len(records)} generations to {args.output}")


if __name__ == "__main__":
    main()
