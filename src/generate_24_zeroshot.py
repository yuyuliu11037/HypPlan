"""Zero-shot and few-shot baselines for Game of 24."""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = (
    "Use the four given numbers and basic arithmetic operations "
    "(+, -, *, /) to obtain 24. Each number must be used exactly once."
)


def load_fewshot_examples(train_path: str, n: int = 3, seed: int = 42) -> list[str]:
    """Load n verified examples from training data for few-shot prompt."""
    with open(train_path) as f:
        lines = [json.loads(l) for l in f]

    # One trajectory per problem
    seen = {}
    for item in lines:
        if item["problem"] not in seen:
            seen[item["problem"]] = item["text"]

    rng = random.Random(seed)
    problems = list(seen.keys())
    rng.shuffle(problems)
    return [seen[p] for p in problems[:n]]


def build_prompt(problem: str, few_shot_examples: list[str] | None = None) -> str:
    nums = problem.replace(",", " ")

    if few_shot_examples:
        parts = [INSTRUCTION + "\n"]
        for ex in few_shot_examples:
            parts.append(ex + "\n")
        parts.append(f"Problem: {nums}\nStep 1:")
        return "\n".join(parts)
    else:
        return f"{INSTRUCTION}\n\nProblem: {nums}\nStep 1:"


def load_model(model_name: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.0, device: str = "cuda") -> str:
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--test_data", default="data/24_test.jsonl")
    parser.add_argument("--train_data", default="data/24_train.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_shots", type=int, default=0,
                        help="Number of few-shot examples (0=zero-shot)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load few-shot examples from train set
    few_shot = None
    if args.num_shots > 0:
        few_shot = load_fewshot_examples(args.train_data, n=args.num_shots)
        print(f"Using {len(few_shot)} few-shot examples")

    # Load test data — deduplicate by problem
    seen = set()
    records = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            if item["problem"] not in seen:
                seen.add(item["problem"])
                records.append(item)

    print(f"Loaded {len(records)} unique problems from {args.test_data}")

    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # Print example prompt
    example_prompt = build_prompt(records[0]["problem"], few_shot)
    print(f"\n--- Example prompt ---\n{example_prompt}\n--- End ---\n")

    with open(args.output, "w") as fout:
        for i, record in enumerate(records):
            prompt = build_prompt(record["problem"], few_shot)
            generation = generate(
                model, tokenizer, prompt,
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
