"""Inference with [PLAN] token hook for HypPlan model."""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.model.proj import ProjMLP, ProjectBack
from src.model.hyperbolic import exp_map_origin


def load_model(config_path: str, stage3_dir: str, device: str = "cuda"):
    """Load the full HypPlan model from Stage 3 checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model"]
    hyp_dim = config["model"]["hyp_dim"]
    proj_hidden_dims = config["model"]["proj_hidden_dims"]

    # Load tokenizer
    tokenizer_path = os.path.join(stage3_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [config["model"]["plan_token"]]}
        )

    plan_token_id = tokenizer.convert_tokens_to_ids(config["model"]["plan_token"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapters
    lora_path = os.path.join(stage3_dir, "lora_adapters")
    if os.path.exists(lora_path):
        base_model = PeftModel.from_pretrained(base_model, lora_path)
        base_model = base_model.merge_and_unload()

    base_model = base_model.to(device).eval()
    hidden_dim = base_model.config.hidden_size

    # Load Proj and ProjectBack
    proj = ProjMLP(hidden_dim, hyp_dim, proj_hidden_dims).to(device)
    project_back = ProjectBack(hyp_dim, hidden_dim).to(device)

    proj_ckpt = os.path.join(stage3_dir, "proj_checkpoint.pt")
    if os.path.exists(proj_ckpt):
        ckpt = torch.load(proj_ckpt, map_location=device, weights_only=True)
        proj.load_state_dict(ckpt["proj"])
        project_back.load_state_dict(ckpt["project_back"])

    proj.eval()
    project_back.eval()

    return base_model, tokenizer, proj, project_back, plan_token_id


@torch.no_grad()
def generate(base_model, tokenizer, proj, project_back, plan_token_id,
             problem: str, max_new_tokens: int = 2048, temperature: float = 0.0,
             device: str = "cuda") -> str:
    """Autoregressive generation with [PLAN] token hook.

    When the model predicts [PLAN]:
    1. Compute planning vector from hidden state at [PLAN] position
    2. Append [PLAN] to sequence
    3. Inject planning vector into the next token's embedding
    """
    input_ids = tokenizer.encode(problem, return_tensors="pt").to(device)
    past_key_values = None
    pending_plan_vector = None

    generated_ids = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        if past_key_values is not None:
            # Only feed the last token for KV-cache inference
            cur_input = torch.tensor([[generated_ids[-1]]], device=device)
        else:
            cur_input = torch.tensor([generated_ids], device=device)

        # Get embeddings
        embeds = base_model.get_input_embeddings()(cur_input)

        # Inject pending plan vector into first position of current input
        if pending_plan_vector is not None:
            delta = project_back(pending_plan_vector)  # (1, embed_dim)
            embeds[:, 0, :] = embeds[:, 0, :] + delta
            pending_plan_vector = None

        outputs = base_model(
            inputs_embeds=embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
        hidden = outputs.hidden_states[-1][:, -1, :]  # (1, H)

        # Sample or greedy
        if temperature <= 0:
            next_token = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == tokenizer.eos_token_id:
            break

        if next_token == plan_token_id:
            # Compute planning vector from hidden state at [PLAN] position
            _, z = proj(hidden)  # z: (1, hyp_dim)
            pending_plan_vector = z  # inject into next token
            generated_ids.append(next_token)
        else:
            generated_ids.append(next_token)

    # Decode, skipping special tokens
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # Remove the problem prefix
    if text.startswith(problem):
        text = text[len(problem):].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage3_dir", default="checkpoints/stage3")
    parser.add_argument("--input", required=True, help="JSONL with 'problem' field")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_model, tokenizer, proj, project_back, plan_token_id = load_model(
        args.config, args.stage3_dir, args.device
    )

    with open(args.input) as f:
        records = [json.loads(line) for line in f]

    results = []
    for i, record in enumerate(records):
        generation = generate(
            base_model, tokenizer, proj, project_back, plan_token_id,
            record["problem"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        results.append({
            "problem": record["problem"],
            "solution": record.get("solution", ""),
            "generation": generation,
            "level": record.get("level", ""),
            "type": record.get("type", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{len(records)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} generations to {args.output}")


if __name__ == "__main__":
    main()
