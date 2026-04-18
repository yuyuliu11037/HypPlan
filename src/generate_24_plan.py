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
             device: str = "cuda", random_z: bool = False,
             z_norm: float = 1.0, z_scale: float = 1.0,
             max_z_injections: int = 3) -> str:
    """Autoregressive generation with planning vector insertion at step boundaries.

    Args:
        z_scale: multiplicative factor applied to z before injection.
        max_z_injections: at most this many z's are injected. With 3, z is injected
            before steps 1, 2, 3. With 2, step-3 injection is skipped (zero vector
            conceptually — we just don't inject).
        random_z: use Gaussian(norm=z_norm) instead of ProjMLP output.
    """
    def make_z(hidden):
        if random_z:
            g = torch.randn_like(hidden)
            g = g * (z_norm / g.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            return g * z_scale
        else:
            _, z = proj(hidden)
            return z * z_scale

    prompt = make_prompt(problem)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Forward the full prompt to prime KV cache
    prompt_out = model(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=True,
    )
    past_key_values = prompt_out.past_key_values
    prompt_logits = prompt_out.logits[:, -1, :]          # predicts token at prompt_len
    prompt_hidden = prompt_out.hidden_states[-1][:, -1, :]  # hidden at last prompt token

    generated_ids = input_ids[0].tolist()
    prompt_len = input_ids.size(1)

    injection_count = 0

    # ── Inject z_step1 (or skip) ──
    # KV already has positions 0..prompt_len-1. Forwarding z alone appends one
    # position (prompt_len) with z's embedding. Logits at that position predict
    # token at prompt_len+1 (first content token of Step 1).
    if injection_count < max_z_injections:
        z_step1 = make_z(prompt_hidden)
        out_z = model(
            inputs_embeds=z_step1.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = out_z.past_key_values
        logits = out_z.logits[:, -1, :]
        # hidden at z position (not used for boundary detection)
        injection_count += 1
    else:
        # No z injection: use prompt_out.logits to predict first content token
        logits = prompt_logits

    # Generate first token
    if temperature <= 0:
        next_token = logits.argmax(dim=-1).item()
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
    if next_token == tokenizer.eos_token_id:
        gen_ids = generated_ids[prompt_len:]
        return "Step 1:" + tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    generated_ids.append(next_token)

    # Boundary pattern: newline followed by "Step" (for steps 2, 3)
    _boundary_re = re.compile(r'\nStep \d+:')
    prev_boundary_count = 0
    pending_plan_vector = None

    for step in range(1, max_new_tokens):
        cur_input = torch.tensor([[generated_ids[-1]]], device=device)
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

        # Check for new step boundary in decoded text (triggers z for NEXT iter)
        tail = tokenizer.decode(
            generated_ids[max(prompt_len, len(generated_ids) - 10):],
            skip_special_tokens=True,
        )
        cur_boundary_count = len(_boundary_re.findall(tail))
        if cur_boundary_count > prev_boundary_count:
            if injection_count < max_z_injections:
                pending_plan_vector = make_z(hidden)
                injection_count += 1
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
    parser.add_argument("--random_z", action="store_true",
                        help="Use random Gaussian z (norm-matched) instead of trained ProjMLP.")
    parser.add_argument("--z_norm", type=float, default=1.0,
                        help="Target norm for random z (matches trained plan_vector_scale).")
    parser.add_argument("--z_scale", type=float, default=1.0,
                        help="Multiplicative scaling applied to z before injection.")
    parser.add_argument("--max_z_injections", type=int, default=3,
                        help="Inject z at most this many times (3=all steps, 2=skip step 3).")
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
                random_z=args.random_z,
                z_norm=args.z_norm,
                z_scale=args.z_scale,
                max_z_injections=args.max_z_injections,
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
