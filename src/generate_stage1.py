"""Inference for Stage 1: autoregressive generation with planning vector insertion."""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.multiprocessing as mp
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.projections import ProjMLP


def load_model(config: dict, checkpoint_dir: str, device: str = "cuda"):
    """Load base model + Stage 1 projection module."""
    model_name = config["model"]["base_model"]
    proj_hidden_dims = config["model"]["proj_hidden_dims"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (frozen, no LoRA in Stage 1)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_model = base_model.to(device).eval()

    hidden_dim = base_model.config.hidden_size

    # Load projection module
    target_norm = config["model"].get("plan_vector_scale", 1.0)
    proj = ProjMLP(hidden_dim, proj_hidden_dims, target_norm=target_norm).to(torch.bfloat16).to(device)
    ckpt = torch.load(
        os.path.join(checkpoint_dir, "checkpoint.pt"),
        map_location=device, weights_only=True,
    )
    proj.load_state_dict(ckpt["proj"])
    proj.eval()

    return base_model, tokenizer, proj


@torch.no_grad()
def generate(base_model, tokenizer, proj,
             problem: str, max_new_tokens: int = 2048, temperature: float = 0.0,
             device: str = "cuda", return_plan_vectors: bool = False):
    """Autoregressive generation with planning vector insertion at step boundaries.

    When a period token is generated, computes a planning vector z from the hidden
    state and inserts it as a virtual token embedding before the next token.

    If return_plan_vectors is True, returns (text, plan_vectors) where plan_vectors
    is a list of dicts with keys 'z' (tensor), 'step' (generation step index),
    and 'context' (text generated up to the boundary).
    """
    input_ids = tokenizer.encode(problem, return_tensors="pt").to(device)
    past_key_values = None
    pending_plan_vector = None

    generated_ids = input_ids[0].tolist()
    prompt_len = input_ids.size(1)
    plan_vectors = []

    # Boundary detection matching training's split_steps regex: (?<!\d)\.(?=\s)
    # The tokenizer may merge periods with adjacent whitespace into single tokens
    # (e.g. ".\n\n" -> token 382), so we check the decoded text rather than token IDs.
    # We track boundary count on a running decoded string to avoid double-triggering.
    import re
    _boundary_re = re.compile(r'(?<!\d)\.\s')
    prev_boundary_count = 0

    for step in range(max_new_tokens):
        if past_key_values is not None:
            cur_input = torch.tensor([[generated_ids[-1]]], device=device)
        else:
            cur_input = torch.tensor([generated_ids], device=device)

        # Get embeddings
        embeds = base_model.get_input_embeddings()(cur_input)

        # If there's a pending plan vector, insert it before the current token
        if pending_plan_vector is not None:
            # pending_plan_vector is (1, H), insert as a virtual token before current
            embeds = torch.cat([pending_plan_vector.unsqueeze(1), embeds], dim=1)
            pending_plan_vector = None

        outputs = base_model(
            inputs_embeds=embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1][:, -1, :]  # (1, H)

        # Sample or greedy
        if temperature <= 0:
            next_token = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        if next_token == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token)

        # Check if a new step boundary appeared in the decoded text.
        # Only decode the last few tokens for efficiency — boundaries span at most
        # 2-3 tokens (e.g. preceding char + ".\n\n").
        tail = tokenizer.decode(generated_ids[max(prompt_len, len(generated_ids) - 5):],
                                skip_special_tokens=True)
        cur_boundary_count = len(_boundary_re.findall(tail))
        if cur_boundary_count > prev_boundary_count:
            _, z = proj(hidden)  # z: (1, H)
            pending_plan_vector = z
            if return_plan_vectors:
                context = tokenizer.decode(generated_ids[prompt_len:],
                                           skip_special_tokens=True)
                plan_vectors.append({
                    "z": z.detach().cpu().squeeze(0),  # (H,)
                    "step": step,
                    "context": context,
                })
        prev_boundary_count = cur_boundary_count

    # Decode only the generated part
    gen_ids = generated_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if return_plan_vectors:
        return text, plan_vectors
    return text


def _worker(rank, world_size, config, args, records):
    """Worker for multi-GPU inference."""
    device = f"cuda:{rank}"
    base_model, tokenizer, proj = load_model(config, args.checkpoint_dir, device)

    shard = [r for i, r in enumerate(records) if i % world_size == rank]

    results = []
    for i, record in enumerate(shard):
        generation = generate(
            base_model, tokenizer, proj,
            record["problem"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        results.append({
            "idx": record["_idx"],
            "problem": record["problem"],
            "solution": record["solution"],
            "generation": generation,
            "level": record.get("level", ""),
            "type": record.get("type", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"[GPU {rank}] Generated {i+1}/{len(shard)}")

    shard_path = args.output + f".shard{rank}"
    with open(shard_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[GPU {rank}] Done — {len(results)} generations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints/stage1")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max test samples (0=use config eval_samples)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load test data
    from datasets import load_dataset, concatenate_datasets
    all_ds = []
    for cfg_name in config["data"]["configs"]:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg_name, split="test")
        all_ds.append(ds)
    test_data = concatenate_datasets(all_ds)

    max_samples = args.max_samples or config["data"].get("eval_samples", len(test_data))
    if max_samples < len(test_data):
        test_data = test_data.shuffle(seed=42).select(range(max_samples))

    records = []
    for i in range(len(test_data)):
        item = test_data[i]
        item["_idx"] = i
        records.append(item)

    num_gpus = args.num_gpus or torch.cuda.device_count()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if num_gpus <= 1:
        _worker(0, 1, config, args, records)
    else:
        print(f"Launching {num_gpus}-GPU parallel inference")
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=_worker, args=(rank, num_gpus, config, args, records))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # Merge shards
    all_results = []
    for rank in range(num_gpus):
        shard_path = args.output + f".shard{rank}"
        if os.path.exists(shard_path):
            with open(shard_path) as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(shard_path)

    all_results.sort(key=lambda r: r["idx"])
    with open(args.output, "w") as f:
        for r in all_results:
            del r["idx"]
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(all_results)} generations to {args.output}")


if __name__ == "__main__":
    main()
