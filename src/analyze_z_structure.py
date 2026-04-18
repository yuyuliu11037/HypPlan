"""Analyze structure of trained ProjMLP output vectors z_i on held-out contexts.

Two diagnostics:
1. PCA spectrum of {z_i}: if effective rank is tiny (e.g. top-1 explains >90%),
   z has collapsed to a near-constant direction.
2. Cosine similarity grouped by:
   - same problem, different steps
   - different problems, same step index
   - different problems, different steps
   If z is problem-informative, same-problem-different-step should differ from
   different-problem-different-step; otherwise z is uncorrelated with problem
   structure.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.projections import ProjMLP
from src.dataset_24_plan import Game24PlanDataset


@torch.no_grad()
def collect_zs(model, proj, dataset, device, max_seq_len):
    """Return list of dicts: {problem, step_idx, z}."""
    records = []
    for sample in dataset:
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        bp = sample["boundary_positions"]
        problem = sample.get("problem") if isinstance(sample, dict) else None

        # Game24PlanDataset doesn't return 'problem' — we need to track via dataset.data
        # Instead, use sample index -> dataset.data[idx]['problem']
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1][0]  # (L, H)

        valid_bp = bp[bp >= 0].tolist()
        for step_idx, pos in enumerate(valid_bp):
            h = hidden_states[pos].unsqueeze(0).float()
            _, z = proj(h)
            records.append({
                "step_idx": step_idx,
                "z": z.squeeze(0).detach().cpu().numpy(),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/sft_24_tot_merged")
    parser.add_argument("--proj_checkpoint", default="checkpoints/plan_24_tot")
    parser.add_argument("--val_data", default="data/24_val_tot.jsonl")
    parser.add_argument("--output", default="results/24_plan_tot/z_structure.json")
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.proj_checkpoint, "config.yaml")) as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    proj = ProjMLP(
        model.config.hidden_size,
        config["model"]["proj_hidden_dims"],
        target_norm=config["model"].get("plan_vector_scale", 1.0),
    )
    proj.load_state_dict(torch.load(
        os.path.join(args.proj_checkpoint, "proj.pt"), map_location=device,
    ))
    proj = proj.to(device).eval()

    dataset = Game24PlanDataset(tokenizer, args.val_data, max_seq_len=args.max_seq_len)
    print(f"Loaded {len(dataset)} trajectories")

    # Collect z's, track problem via dataset.data
    records = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        problem = dataset.data[idx]["problem"]
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        bp = sample["boundary_positions"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1][0]

        valid_bp = bp[bp >= 0].tolist()
        for step_idx, pos in enumerate(valid_bp):
            h = hidden_states[pos].unsqueeze(0).float()
            _, z = proj(h)
            records.append({
                "problem": problem,
                "step_idx": step_idx,
                "z": z.squeeze(0).detach().cpu().numpy(),
            })

        if (idx + 1) % 100 == 0:
            print(f"  collected from {idx+1}/{len(dataset)} trajectories, total z's: {len(records)}")

    # Assemble Z matrix (N, H)
    Z = np.stack([r["z"] for r in records])
    problems = np.array([r["problem"] for r in records])
    step_idx = np.array([r["step_idx"] for r in records])
    print(f"\nTotal z vectors: {Z.shape[0]}, dim: {Z.shape[1]}")
    print(f"z norms: mean={np.linalg.norm(Z, axis=1).mean():.4f}, "
          f"std={np.linalg.norm(Z, axis=1).std():.4f}")

    # ── PCA spectrum ──────────────────────────────────────────────────────────
    # Center
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
    var = S ** 2
    var_explained = var / var.sum()
    cumvar = np.cumsum(var_explained)

    # How many PCs to reach 50/90/99% variance
    def pcs_to_reach(threshold):
        return int(np.searchsorted(cumvar, threshold) + 1)

    pca_spec = {
        "top1_var_explained": float(var_explained[0]),
        "top5_var_explained": float(cumvar[min(4, len(cumvar)-1)]),
        "top10_var_explained": float(cumvar[min(9, len(cumvar)-1)]),
        "top50_var_explained": float(cumvar[min(49, len(cumvar)-1)]),
        "pcs_for_50pct": pcs_to_reach(0.50),
        "pcs_for_90pct": pcs_to_reach(0.90),
        "pcs_for_99pct": pcs_to_reach(0.99),
        "effective_rank_nuclear": float(S.sum() / S.max()),
    }
    print("\n=== PCA Spectrum ===")
    for k, v in pca_spec.items():
        print(f"  {k}: {v}")

    # ── Cosine similarities ───────────────────────────────────────────────────
    # Normalize
    Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    # Build categories by sampling pairs
    N = len(records)
    rng = np.random.RandomState(42)
    max_pairs = 20000  # cap pairs per category

    same_prob_diff_step = []
    diff_prob_same_step = []
    diff_prob_diff_step = []

    # Efficient: build index
    from collections import defaultdict
    by_problem = defaultdict(list)
    by_step = defaultdict(list)
    for i, (p, s) in enumerate(zip(problems, step_idx)):
        by_problem[p].append(i)
        by_step[s].append(i)

    # Same problem, different step: iterate pairs within problem with different step
    for p, idxs in by_problem.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                if step_idx[idxs[i]] != step_idx[idxs[j]]:
                    same_prob_diff_step.append(
                        float(Z_norm[idxs[i]] @ Z_norm[idxs[j]])
                    )
            if len(same_prob_diff_step) >= max_pairs:
                break
        if len(same_prob_diff_step) >= max_pairs:
            break

    # Different problems, same step: sample pairs within step with different problem
    for s, idxs in by_step.items():
        rng.shuffle(idxs)
        for i in range(len(idxs)):
            for j in range(i + 1, min(i + 20, len(idxs))):  # cap inner
                if problems[idxs[i]] != problems[idxs[j]]:
                    diff_prob_same_step.append(
                        float(Z_norm[idxs[i]] @ Z_norm[idxs[j]])
                    )
            if len(diff_prob_same_step) >= max_pairs:
                break
        if len(diff_prob_same_step) >= max_pairs:
            break

    # Different problems, different steps: random pairs
    for _ in range(max_pairs):
        i, j = rng.randint(0, N, size=2)
        if problems[i] != problems[j] and step_idx[i] != step_idx[j]:
            diff_prob_diff_step.append(
                float(Z_norm[i] @ Z_norm[j])
            )

    def stats(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {
            "n": int(len(a)),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "median": float(np.median(a)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    cos_stats = {
        "same_problem_diff_step": stats(same_prob_diff_step),
        "diff_problem_same_step": stats(diff_prob_same_step),
        "diff_problem_diff_step": stats(diff_prob_diff_step),
    }
    print("\n=== Cosine Similarity ===")
    for k, v in cos_stats.items():
        print(f"  {k}: n={v['n']}, mean={v.get('mean', 'N/A'):.4f}, "
              f"std={v.get('std', 'N/A'):.4f}")

    # Mean direction alignment: how close is everything to the overall mean?
    mean_z = Z_norm.mean(axis=0)
    mean_z_norm = mean_z / (np.linalg.norm(mean_z) + 1e-8)
    sim_to_mean = Z_norm @ mean_z_norm
    cos_stats["similarity_to_mean_direction"] = stats(sim_to_mean.tolist())
    print(f"  similarity_to_mean_direction: n={len(sim_to_mean)}, "
          f"mean={sim_to_mean.mean():.4f}, std={sim_to_mean.std():.4f}")

    # Save
    output = {
        "n_vectors": int(N),
        "hidden_dim": int(Z.shape[1]),
        "z_norm_mean": float(np.linalg.norm(Z, axis=1).mean()),
        "z_norm_std": float(np.linalg.norm(Z, axis=1).std()),
        "pca_spectrum": pca_spec,
        "top_20_singular_values": S[:20].tolist(),
        "cosine_stats": cos_stats,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
