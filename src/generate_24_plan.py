"""Inference with planning vectors for Game of 24.

Loads merged SFT model + trained ProjMLP. During autoregressive generation,
detects step boundaries and inserts planning vectors as virtual tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.projections import ProjMLP
from src.dataset_24 import make_prompt


def load_model(base_model_path: str, proj_checkpoint_dir: str, device: str = "cuda"):
    """Load base model + ProjMLP."""
    # Load config from proj checkpoint
    with open(os.path.join(proj_checkpoint_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    hidden_dim = model.config.hidden_size
    proj = ProjMLP(
        hidden_dim,
        config["model"]["proj_hidden_dims"],
        target_norm=config["model"].get("plan_vector_scale", 1.0),
    )
    proj.load_state_dict(torch.load(
        os.path.join(proj_checkpoint_dir, "proj.pt"),
        map_location=device,
    ))
    proj = proj.to(device).to(torch.bfloat16).eval()

    return model, tokenizer, proj


@torch.no_grad()
def generate(model, tokenizer, proj, problem: str,
             max_new_tokens: int = 256, temperature: float = 0.0,
             device: str = "cuda") -> str:
    """Autoregressive generation with planning vector insertion at step boundaries.

    Boundary detection: looks for newline + "Step N:" pattern in decoded text,
    matching the Game of 24 trajectory format.
    """
    prompt = make_prompt(problem)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None
    pending_plan_vector = None

    generated_ids = input_ids[0].tolist()
    prompt_len = input_ids.size(1)

    # Boundary pattern: newline followed by "Step" (steps 2, 3, ...)
    _boundary_re = re.compile(r'\nStep \d+:')
    prev_boundary_count = 0

    for step in range(max_new_tokens):
        if past_key_values is not None:
            cur_input = torch.tensor([[generated_ids[-1]]], device=device)
        else:
            cur_input = torch.tensor([generated_ids], device=device)

        embeds = model.get_input_embeddings()(cur_input)

        if pending_plan_vector is not None:
            embeds = torch.cat([pending_plan_vector.unsqueeze(1), embeds], dim=1)
            pending_plan_vector = None

        outputs = model(
            inputs_embeds=embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1][:, -1, :]  # (1, H)

        if temperature <= 0:
            next_token = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token)

        # Check for new step boundary in decoded text
        tail = tokenizer.decode(
            generated_ids[max(prompt_len, len(generated_ids) - 10):],
            skip_special_tokens=True,
        )
        cur_boundary_count = len(_boundary_re.findall(tail))
        if cur_boundary_count > prev_boundary_count:
            _, z = proj(hidden)  # z: (1, H)
            pending_plan_vector = z
        prev_boundary_count = cur_boundary_count

    gen_ids = generated_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return "Step 1:" + text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft_24_v2_merged")
    parser.add_argument("--proj_checkpoint", default="checkpoints/plan_24")
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

    model, tokenizer, proj = load_model(args.base_model, args.proj_checkpoint)
    device = next(model.parameters()).device

    with open(args.output, "w") as fout:
        for i, record in enumerate(records):
            generation = generate(
                model, tokenizer, proj, record["problem"],
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
