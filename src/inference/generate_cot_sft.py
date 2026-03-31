"""Inference for CoT-SFT baseline: standard generation without [PLAN] hook."""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(config_path: str, checkpoint_dir: str, device: str = "cuda"):
    """Load CoT-SFT model (base + LoRA adapters)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model"]

    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + merge LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_path = os.path.join(checkpoint_dir, "lora_adapters")
    if os.path.exists(lora_path):
        base_model = PeftModel.from_pretrained(base_model, lora_path)
        base_model = base_model.merge_and_unload()

    base_model = base_model.to(device).eval()
    return base_model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, problem: str, max_new_tokens: int = 2048,
             temperature: float = 0.0, device: str = "cuda") -> str:
    """Standard autoregressive generation (no planning tokens)."""
    input_ids = tokenizer.encode(problem, return_tensors="pt").to(device)

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

    # Decode only the generated part
    generated_ids = output_ids[0, input_ids.size(1):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _worker(rank, world_size, args, records):
    """Worker function for multi-GPU inference."""
    device = f"cuda:{rank}"
    model, tokenizer = load_model(args.config, args.checkpoint_dir, device)

    # Split data by rank
    shard = [r for i, r in enumerate(records) if i % world_size == rank]

    results = []
    for i, record in enumerate(shard):
        generation = generate(
            model, tokenizer, record["problem"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        results.append({
            "idx": record["_idx"],
            "problem": record["problem"],
            "solution": record.get("solution", ""),
            "generation": generation,
            "level": record.get("level", ""),
            "type": record.get("type", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"[GPU {rank}] Generated {i+1}/{len(shard)}")

    # Write shard results to a temp file
    shard_path = args.output + f".shard{rank}"
    with open(shard_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[GPU {rank}] Done — {len(results)} generations")


def main():
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints/cot_sft")
    parser.add_argument("--input", required=True, help="JSONL with 'problem' field")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs for parallel inference (0=auto)")
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f]

    # Tag each record with original index for ordering
    for i, r in enumerate(records):
        r["_idx"] = i

    num_gpus = args.num_gpus or torch.cuda.device_count()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if num_gpus <= 1:
        _worker(0, 1, args, records)
    else:
        print(f"Launching {num_gpus}-GPU parallel inference")
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=_worker, args=(rank, num_gpus, args, records))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # Merge shards in original order
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
