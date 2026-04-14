"""Inference for CoT-SFT model on Game of 24."""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.dataset_24 import make_prompt


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
def generate(model, tokenizer, problem: str, max_new_tokens: int = 256,
             temperature: float = 0.0, device: str = "cuda") -> str:
    prompt = make_prompt(problem)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if temperature <= 0:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_ids.size(1):]
    return "Step 1:" + tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--test_data", default="data/24_test.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Deduplicate by problem
    seen = set()
    records = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item["problem"] not in seen:
                seen.add(item["problem"])
                records.append(item)

    print(f"Loaded {len(records)} unique problems from {args.test_data}")

    model, tokenizer = load_model(args.base_model, args.adapter)
    device = next(model.parameters()).device

    with open(args.output, "w") as fout:
        for i, record in enumerate(records):
            generation = generate(
                model, tokenizer, record["problem"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )
            result = {
                "problem": record["problem"],
                "ground_truth": record["text"],
                "generation": generation,
            }
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            if (i + 1) % 20 == 0:
                print(f"Generated {i+1}/{len(records)}")

    print(f"Saved {len(records)} generations to {args.output}")


if __name__ == "__main__":
    main()
