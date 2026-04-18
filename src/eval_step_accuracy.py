"""Step-level next-step accuracy: with vs without planning vector z.

On held-out val trajectories, at each step boundary, greedy-generate the next
step's tokens from the frozen SFT model both WITH and WITHOUT z injected, and
compare against the ground-truth step. Reports exact-step-match accuracy and
token-level accuracy across all boundaries.

This isolates what the ProjMLP learned to do, without letting downstream
compounding errors dilute the signal.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.projections import ProjMLP
from src.dataset_24_plan import Game24PlanDataset


@torch.no_grad()
def greedy_generate(model, prefix_embeds, z_i, target_len, device):
    """Greedy-decode target_len tokens starting from prefix (+ optional z).

    Args:
        prefix_embeds: (L, H) prefix embeddings (NO batch dim).
        z_i: (H,) planning vector or None.
        target_len: number of tokens to generate.

    Returns:
        list[int]: predicted token ids of length target_len.
    """
    if z_i is not None:
        full = torch.cat([prefix_embeds, z_i.unsqueeze(0)], dim=0).unsqueeze(0)
    else:
        full = prefix_embeds.unsqueeze(0)

    past_kv = None
    cur_embeds = full
    embed_table = model.get_input_embeddings()
    out_ids = []
    for _ in range(target_len):
        out = model(inputs_embeds=cur_embeds, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1).item()
        out_ids.append(next_id)
        cur_embeds = embed_table(torch.tensor([[next_id]], device=device))
    return out_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft_24_tot_merged")
    parser.add_argument("--proj_checkpoint", default="checkpoints/plan_24_tot")
    parser.add_argument("--val_data", default="data/24_val_tot.jsonl")
    parser.add_argument("--output", default="results/24_plan_tot/step_accuracy.json")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_boundaries", type=int, default=0,
                        help="0 = no cap; otherwise limit total boundaries evaluated.")
    parser.add_argument("--z_scale", type=float, default=1.0,
                        help="Multiplicative factor applied to z before injection.")
    parser.add_argument("--skip_step3", action="store_true",
                        help="At the Step 3 boundary (i=2, last step), don't inject z.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(os.path.join(args.proj_checkpoint, "config.yaml")) as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading frozen SFT model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    proj = ProjMLP(
        hidden_dim,
        config["model"]["proj_hidden_dims"],
        target_norm=config["model"].get("plan_vector_scale", 1.0),
    )
    proj.load_state_dict(torch.load(
        os.path.join(args.proj_checkpoint, "proj.pt"), map_location=device,
    ))
    proj = proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad = False

    dataset = Game24PlanDataset(tokenizer, args.val_data, max_seq_len=args.max_seq_len)
    print(f"Loaded {len(dataset)} val trajectories")

    embed_table = model.get_input_embeddings()

    total_boundaries = 0
    exact_with_z = 0
    exact_without_z = 0
    tok_with_z = 0
    tok_without_z = 0
    tok_total = 0

    for sample_idx, sample in enumerate(dataset):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        boundary_positions = sample["boundary_positions"]

        # Get hidden states from full sequence (frozen, no grad)
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1][0]  # (L, H)

        real_len = attention_mask.sum().item()
        valid_bp = boundary_positions[boundary_positions >= 0].tolist()
        K = len(valid_bp)

        for i in range(K):
            bpos = valid_bp[i]
            step_start = bpos + 1
            step_end = valid_bp[i + 1] + 1 if i < K - 1 else real_len
            if step_start >= step_end:
                continue
            true_step_ids = input_ids[step_start:step_end].tolist()
            target_len = len(true_step_ids)

            # Prefix embeds (everything up to and including the boundary token)
            with torch.no_grad():
                prefix_embeds = embed_table(input_ids[:bpos + 1]).to(torch.bfloat16)

            # Compute z from hidden state at bpos
            h_i = hidden_states[bpos].unsqueeze(0).float()
            _, z_i = proj(h_i)
            z_i = z_i * args.z_scale  # amplification ablation

            # Optionally skip z at the LAST boundary (i == K-1 => about to predict step 3)
            use_z = not (args.skip_step3 and i == K - 1)
            z_i_bf = z_i.squeeze(0).to(torch.bfloat16) if use_z else None

            # Predict WITHOUT z
            pred_no_z = greedy_generate(model, prefix_embeds, None, target_len, device)
            # Predict WITH z
            pred_with_z = greedy_generate(model, prefix_embeds, z_i_bf, target_len, device)

            # Compare
            total_boundaries += 1
            if pred_no_z == true_step_ids:
                exact_without_z += 1
            if pred_with_z == true_step_ids:
                exact_with_z += 1

            # Token-level
            for pt, tt in zip(pred_no_z, true_step_ids):
                if pt == tt:
                    tok_without_z += 1
            for pt, tt in zip(pred_with_z, true_step_ids):
                if pt == tt:
                    tok_with_z += 1
            tok_total += target_len

            if args.max_boundaries and total_boundaries >= args.max_boundaries:
                break
        if args.max_boundaries and total_boundaries >= args.max_boundaries:
            break

        if (sample_idx + 1) % 50 == 0:
            print(f"  [{sample_idx+1}/{len(dataset)}] boundaries={total_boundaries} "
                  f"exact_with={exact_with_z} exact_without={exact_without_z}")

    results = {
        "total_boundaries": total_boundaries,
        "exact_step_match": {
            "with_z": exact_with_z,
            "without_z": exact_without_z,
            "acc_with_z": exact_with_z / max(total_boundaries, 1),
            "acc_without_z": exact_without_z / max(total_boundaries, 1),
            "delta": (exact_with_z - exact_without_z) / max(total_boundaries, 1),
        },
        "token_level": {
            "total_tokens": tok_total,
            "with_z": tok_with_z,
            "without_z": tok_without_z,
            "acc_with_z": tok_with_z / max(tok_total, 1),
            "acc_without_z": tok_without_z / max(tok_total, 1),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Total boundaries: {total_boundaries}")
    print(f"Exact step-match accuracy:")
    print(f"  with z:    {results['exact_step_match']['acc_with_z']:.4f} "
          f"({exact_with_z}/{total_boundaries})")
    print(f"  without z: {results['exact_step_match']['acc_without_z']:.4f} "
          f"({exact_without_z}/{total_boundaries})")
    print(f"  delta:     {results['exact_step_match']['delta']:+.4f}")
    print(f"Token-level accuracy:")
    print(f"  with z:    {results['token_level']['acc_with_z']:.4f}")
    print(f"  without z: {results['token_level']['acc_without_z']:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
