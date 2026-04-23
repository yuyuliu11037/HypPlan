"""Stage-2 inference: LoRA + frozen head + trainable-then-loaded UpProjector.

At each detected step boundary during autoregressive generation, reconstruct
the canonical state text (problem + ops so far), forward through the base
with LoRA disabled to get a frozen SFT hidden, push through the frozen head
and the loaded up-projector, and inject the result as a virtual token before
the next step.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.head import HyperbolicHead, UpProjector
from src.prompt_builders import get_builder_cd, sft_prompt_cd
from src.dataset_24_stage2 import STEP_RE
from src.tree_data_cd import render_state_from_history


_boundary_re = re.compile(r"\nStep \d+:")


def history_from_generation(gen_text: str) -> tuple:
    """Parse Countdown integer-arithmetic steps from generated text."""
    hist = []
    for m in STEP_RE.finditer(gen_text):
        try:
            a = int(m.group(1))
            op = m.group(2)
            b = int(m.group(3))
            r = int(m.group(4).rstrip("."))
            hist.append((a, op, b, r))
        except (ValueError, TypeError):
            continue
    return tuple(hist)


def load_all(
    base_model_path: str,
    lora_path: str,
    head_path: str,
    up_proj_path: str,
    stage2_config: dict,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.to(device).eval()

    head_ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    head_cfg = head_ckpt["config"]["model"]
    head = HyperbolicHead(
        in_dim=head_ckpt["in_dim"],
        hyp_dim=head_cfg["hyp_dim"],
        hidden_dims=head_cfg["head_hidden_dims"],
        manifold=head_cfg["manifold"],
    ).to(device).float()
    head.load_state_dict(head_ckpt["state_dict"])
    head.eval()

    up_in = head_cfg["hyp_dim"] + (1 if head_cfg["manifold"] == "lorentz" else 0)
    up_proj = UpProjector(
        in_dim=up_in,
        hidden=stage2_config["model"]["up_proj_hidden"],
        out_dim=base.config.hidden_size,
    ).to(device).float()
    up_proj.load_state_dict(torch.load(up_proj_path, map_location=device,
                                        weights_only=True))
    up_proj.eval()

    return model, tokenizer, head, up_proj


@torch.no_grad()
def compute_z_inj(model, tokenizer, head, up_proj,
                   pool: list, target: int, history: tuple,
                   device, random_z: bool = False) -> torch.Tensor:
    """Render canonical Countdown state, get hidden from base (LoRA disabled),
    head + up_proj."""
    if random_z:
        hidden_dim = head.mlp[0].in_features
        g = torch.randn(1, hidden_dim, device=device)
        return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    state_text = render_state_from_history(list(pool), target, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True, return_tensors="pt").to(device)
    with model.disable_adapter():
        out = model(input_ids=ids, output_hidden_states=True)
        last_h = out.hidden_states[-1][:, -1, :]
    z_hyp = head(last_h.float())
    z_inj = up_proj(z_hyp)
    return z_inj


@torch.no_grad()
def generate(model, tokenizer, head, up_proj,
              pool: list, target: int,
              max_new_tokens: int = 512, temperature: float = 0.0,
              device: str = "cuda", random_z: bool = False,
              max_z_injections: int = 5, no_z_inject: bool = False,
              prompt_builder=None) -> str:
    """Autoregressive Countdown generation with optional z-injection."""
    if prompt_builder is None:
        prompt_builder = sft_prompt_cd
    prompt_text, add_special = prompt_builder(tokenizer, pool, target)
    input_ids = tokenizer.encode(
        prompt_text, add_special_tokens=add_special, return_tensors="pt",
    ).to(device)

    prompt_out = model(input_ids=input_ids, use_cache=True)
    past = prompt_out.past_key_values
    logits = prompt_out.logits[:, -1, :]

    generated_ids = input_ids[0].tolist()
    prompt_len = input_ids.size(1)

    injection_count = 0
    embed_table = model.get_input_embeddings()

    # z before Step 1 uses empty history
    if not no_z_inject and injection_count < max_z_injections:
        z1 = compute_z_inj(model, tokenizer, head, up_proj, pool, target,
                            history=tuple(), device=device, random_z=random_z)
        out_z = model(inputs_embeds=z1.unsqueeze(1).to(next(model.parameters()).dtype),
                       past_key_values=past, use_cache=True)
        past = out_z.past_key_values
        logits = out_z.logits[:, -1, :]
        injection_count += 1

    if temperature <= 0:
        next_token = int(logits.argmax(dim=-1).item())
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = int(torch.multinomial(probs, 1).item())
    if next_token == tokenizer.eos_token_id:
        return "Step 1:" + tokenizer.decode(generated_ids[prompt_len:],
                                              skip_special_tokens=True).rstrip()
    generated_ids.append(next_token)

    prev_boundary_count = 0
    pending_z = None

    for _ in range(max_new_tokens - 1):
        cur = torch.tensor([[generated_ids[-1]]], device=device)
        embeds = embed_table(cur)
        if pending_z is not None:
            embeds = torch.cat([
                pending_z.unsqueeze(1).to(embeds.dtype), embeds,
            ], dim=1)
            pending_z = None

        out = model(inputs_embeds=embeds, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        if temperature <= 0:
            next_token = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = int(torch.multinomial(probs, 1).item())

        if next_token == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token)

        # Detect new boundary
        tail_start = max(prompt_len, len(generated_ids) - 12)
        tail = tokenizer.decode(generated_ids[tail_start:], skip_special_tokens=True)
        cur_count = len(_boundary_re.findall(tail))
        if cur_count > prev_boundary_count:
            if not no_z_inject and injection_count < max_z_injections:
                # Parse full history from generated text so far
                full_gen = tokenizer.decode(generated_ids[prompt_len:],
                                             skip_special_tokens=True)
                # The generated text starts mid-Step-1 (prompt ended with "Step 1:").
                # Prepend "Step 1:" so STEP_RE matches.
                hist_text = "Step 1:" + full_gen
                hist = history_from_generation(hist_text)
                pending_z = compute_z_inj(model, tokenizer, head, up_proj,
                                           pool, target, history=hist, device=device,
                                           random_z=random_z)
                injection_count += 1
            prev_boundary_count = cur_count

    gen_ids = generated_ids[prompt_len:]
    return "Step 1:" + tokenizer.decode(gen_ids, skip_special_tokens=True).rstrip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_checkpoint", required=True,
                        help="Directory containing lora/ and up_projector.pt + config.yaml")
    parser.add_argument("--test_data", default="data/24_test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random_z", action="store_true")
    parser.add_argument("--no_z_inject", action="store_true",
                        help="Disable z injection entirely (use for the DAgger no-z arm).")
    parser.add_argument("--max_z_injections", type=int, default=3)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    ckpt_dir = Path(args.stage2_checkpoint)
    with open(ckpt_dir / "config.yaml") as f:
        s2cfg = yaml.safe_load(f)

    # Honor prompt_style from the training config so eval uses the same
    # prompt distribution the LoRA was trained under.
    prompt_style = str(s2cfg.get("training", {}).get("prompt_style", "sft"))
    prompt_builder = get_builder_cd(prompt_style)
    print(f"prompt_style={prompt_style}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    seen = set()
    records = []
    with open(args.test_data) as f:
        for line in f:
            item = json.loads(line)
            key = (tuple(sorted(item["pool"])), item["target"])
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
    if args.limit > 0:
        records = records[: args.limit]
    print(f"Loaded {len(records)} unique problems from {args.test_data}")

    model, tokenizer, head, up_proj = load_all(
        s2cfg["model"]["base_model"],
        str(ckpt_dir / "lora"),
        s2cfg["model"]["head_checkpoint"],
        str(ckpt_dir / "up_projector.pt"),
        s2cfg,
    )
    device = next(model.parameters()).device

    with open(args.output, "w") as fout:
        for i, record in enumerate(records):
            gen = generate(
                model, tokenizer, head, up_proj,
                record["pool"], record["target"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
                random_z=args.random_z,
                no_z_inject=args.no_z_inject,
                max_z_injections=args.max_z_injections,
                prompt_builder=prompt_builder,
            )
            out = {"pool": record["pool"], "target": record["target"],
                   "problem_idx": record.get("problem_idx"),
                   "ground_truth": record.get("text"),
                   "generation": gen}
            fout.write(json.dumps(out) + "\n"); fout.flush()
            if (i + 1) % 20 == 0:
                print(f"Generated {i+1}/{len(records)}")
    print(f"Saved {len(records)} generations to {args.output}")


if __name__ == "__main__":
    main()
