"""Generate 16 solutions per MATH training problem using Qwen2.5-Math-7B-Instruct via vllm."""

import json
import os

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_PATH = "results/math_samples.jsonl"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

NUM_SAMPLES = 16
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 2048


def load_math_train():
    """Load all MATH training subjects and concatenate."""
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    datasets = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        datasets.append(ds)
    combined = concatenate_datasets(datasets)
    print(f"Loaded {len(combined)} MATH training problems")
    return combined


def build_prompts(dataset, tokenizer):
    """Build chat-formatted prompts for each problem."""
    prompts = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dataset = load_math_train()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = build_prompts(dataset, tokenizer)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=NUM_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    print(f"Generating {NUM_SAMPLES} solutions for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    with open(OUTPUT_PATH, "w") as f:
        for i, output in enumerate(outputs):
            example = dataset[i]
            generations = [o.text for o in output.outputs]
            record = {
                "problem": example["problem"],
                "solution": example["solution"],
                "level": example["level"],
                "type": example["type"],
                "generations": generations,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(outputs)} problems with {NUM_SAMPLES} generations each to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
