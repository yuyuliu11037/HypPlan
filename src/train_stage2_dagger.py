"""Stage-2 DAgger trainer: two-arm (`--use_z` on/off) training with oracle
labels from the enumerated Game-of-24 tree.

Per epoch:
  1. **Rollout phase.** For each training problem, generate a trajectory with
     the current policy (temperature + top-p sampling). At each step boundary,
     record the (problem, history, winning_ops) tuple. Truncate at first
     invalid step. Drop the trajectory from that point onward.
  2. **Training phase.** For each collected tuple, pick one winner
     deterministically and teacher-force-CE the model to emit its full step
     text. Backprop into LoRA + UpProjector; head and base remain frozen.

Loss (phase 1): **single-winner CE** on the full step text.
  - Oracle returns the set of winning next-ops. We pick ONE deterministically
    (lex-order on (op_sym, a, b)) and teacher-force-CE the model to emit its
    full step text (operand, operator, operand, result, remaining).
  - Rationale: all winners at a given step share v(s') = len(s')-1 on a
    shallow 3-step tree, so v-based tiebreaking doesn't differentiate. Lex is
    as principled as anything else and keeps the implementation to one forward
    pass per training example.
  - Phase 2 upgrade path: log-of-sum over full step texts of ALL winners via
    gradient-accumulated softmax reweighting (≈K× compute).

Warm start: fresh LoRA (PEFT default, so delta ≈ 0 at step 0) + small-std
init on UpProjector final layer (avoids the LayerNorm(0) edge case that
triggers chat-template fallback). Frozen head checkpoint comes from an
origin_ranking stage-1 run.

Single-GPU first. DDP can be added once the single-GPU pipeline is validated.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import os

import numpy as np
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.dagger_rollout import rollout_one
from src.dataset_24 import make_prompt
from src.head import HyperbolicHead, UpProjector
from src.tree_data import fraction_to_str, render_state_from_history


def load_problems(jsonl_path: str) -> list[str]:
    seen, out = set(), []
    with open(jsonl_path) as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def load_base_and_head(base_path: str, head_path: str, up_proj_hidden: int,
                        lora_r: int, lora_alpha: int, lora_dropout: float,
                        device: torch.device):
    """Load tokenizer + SFT-merged base, attach fresh LoRA (B=0), load frozen
    head, create small-std-init UpProjector."""
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)

    # Frozen head from origin_ranking checkpoint
    sd = torch.load(head_path, map_location="cpu", weights_only=False)
    head_cfg = sd["config"]["model"]
    head = HyperbolicHead(
        in_dim=sd["in_dim"], hyp_dim=head_cfg["hyp_dim"],
        hidden_dims=head_cfg["head_hidden_dims"],
        manifold=head_cfg["manifold"],
    ).to(device).float()
    head.load_state_dict(sd["state_dict"])
    for p in head.parameters():
        p.requires_grad = False
    head.eval()

    up_in = head_cfg["hyp_dim"] + (1 if head_cfg["manifold"] == "lorentz" else 0)
    up_proj = UpProjector(in_dim=up_in, hidden=up_proj_hidden,
                          out_dim=base.config.hidden_size).to(device).float()
    # Small-std init on the final Linear so z_inj starts small but non-zero,
    # avoiding LayerNorm(0) chat-template fallback.
    for m in reversed(list(up_proj.net.modules())):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            torch.nn.init.zeros_(m.bias)
            break

    return tokenizer, model, head, up_proj


def _compute_z_with_grad(model, tokenizer, head, up_proj, problem: str,
                          history: tuple, device: torch.device) -> torch.Tensor:
    """z_inj = up_proj(head(frozen_base(canonical_state_text))).

    Base forward under adapter-disabled no_grad. Head under no_grad (frozen).
    UpProjector with grad (trainable)."""
    state_text = render_state_from_history(problem, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.no_grad():
        with model.disable_adapter():
            out = model(input_ids=ids, output_hidden_states=True)
            last_h = out.hidden_states[-1][:, -1, :]
        z_hyp = head(last_h.float())
    return up_proj(z_hyp)   # (1, H) with grad through up_proj


def _pick_winner(winning_ops):
    """Deterministically pick one winner by lex-order on (op_sym, a, b)."""
    if not winning_ops:
        return None
    return min(winning_ops, key=lambda w: (w[0], float(w[1]), float(w[2])))


def _format_winner_target(a: Fraction, op_sym: str, b: Fraction, r: Fraction,
                           pool_after: list, max_steps: int,
                           step_num: int) -> str:
    """Target text that follows the `Step N:` (and optional z) context.

    Includes the leading space, the op, the result, and either the
    'Remaining: ...' tail (if more steps follow) or 'Answer: 24' (if this is
    the final step producing the target).
    """
    head = f" {fraction_to_str(a)} {op_sym} {fraction_to_str(b)} = {fraction_to_str(r)}"
    if len(pool_after) == 1 and pool_after[0] == Fraction(24):
        return head + ". Answer: 24"
    rem = " ".join(fraction_to_str(x) for x in sorted(pool_after))
    tail = f". Remaining: {rem}"
    next_step = step_num + 1
    if next_step <= max_steps:
        tail += f"\nStep {next_step}:"
    return head + tail


def dagger_loss(model, tokenizer, head, up_proj, problem: str, history: tuple,
                 winning_ops, use_z: bool, device: torch.device,
                 max_steps: int = 3) -> torch.Tensor:
    """Single-winner CE on the full step text of the deterministically-picked
    winner.
    """
    winner = _pick_winner(winning_ops)
    if winner is None:
        return None
    wop, wa, wb, wr = winner

    embed_table = model.get_input_embeddings()
    dtype = next(model.parameters()).dtype

    def _embed_text(text: str, add_special: bool):
        ids = tokenizer.encode(text, add_special_tokens=add_special, return_tensors="pt").to(device)
        return embed_table(ids), ids.size(1)

    pieces = []
    ctx_prompt_embed, _ = _embed_text(make_prompt(problem), add_special=True)
    pieces.append(ctx_prompt_embed)

    pool = [Fraction(int(x)) for x in problem.split(",")]

    for i, (a, op_sym, b, r) in enumerate(history):
        if use_z:
            z = _compute_z_with_grad(model, tokenizer, head, up_proj, problem,
                                      history[:i], device)
            pieces.append(z.unsqueeze(1).to(dtype))
        pool.remove(a); pool.remove(b); pool.append(r)
        remaining_sorted = sorted(pool)
        content = (
            f" {fraction_to_str(a)} {op_sym} {fraction_to_str(b)} = "
            f"{fraction_to_str(r)}. Remaining: "
            + " ".join(fraction_to_str(x) for x in remaining_sorted)
        )
        next_step_num = i + 2
        if next_step_num <= max_steps:
            content += f"\nStep {next_step_num}:"
        emb, _ = _embed_text(content, add_special=False)
        pieces.append(emb)

    if use_z:
        z = _compute_z_with_grad(model, tokenizer, head, up_proj, problem,
                                  history, device)
        pieces.append(z.unsqueeze(1).to(dtype))

    context = torch.cat(pieces, dim=1)
    context_len = context.size(1)

    # Construct target text for the picked winner
    pool_after_winner = list(pool)
    pool_after_winner.remove(wa); pool_after_winner.remove(wb)
    pool_after_winner.append(wr)
    step_num = len(history) + 1
    target_text = _format_winner_target(wa, wop, wb, wr, pool_after_winner,
                                         max_steps, step_num)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False,
                                    return_tensors="pt").to(device)
    target_embeds = embed_table(target_ids)

    combined = torch.cat([context, target_embeds], dim=1)

    # Labels: -100 for context positions (ignored), target ids for target positions.
    # HuggingFace's causal LM auto-shifts labels internally.
    pad_ids = torch.full((1, context_len), -100, dtype=torch.long, device=device)
    labels = torch.cat([pad_ids, target_ids], dim=1)

    out = model(inputs_embeds=combined, labels=labels, use_cache=False)
    return out.loss


def collect_training_pairs(model, tokenizer, head, up_proj, problems,
                             use_z: bool, device, temperature: float, top_p: float,
                             rollouts_per_problem: int, max_new_tokens: int,
                             max_steps: int, log_every: int = 100):
    """Rollout phase. Returns list of (problem, history_before, winning_ops)
    training tuples and rollout-level stats."""
    model.eval()
    pairs = []
    stats = {"n_rollouts": 0, "n_solved": 0,
             "stopped_reason": {}, "dropped_at_step": {},
             "n_boundaries_total": 0, "n_boundaries_invalid": 0,
             "n_boundaries_empty_oracle": 0}

    for pi, problem in enumerate(problems):
        for ri in range(rollouts_per_problem):
            r = rollout_one(model, tokenizer, head, up_proj, problem, device,
                            use_z=use_z, temperature=temperature, top_p=top_p,
                            max_new_tokens=max_new_tokens, max_steps=max_steps)
            stats["n_rollouts"] += 1
            stats["n_solved"] += int(r.solved)
            stats["stopped_reason"][r.stopped_reason] = (
                stats["stopped_reason"].get(r.stopped_reason, 0) + 1)

            for bdy in r.boundaries:
                stats["n_boundaries_total"] += 1
                if bdy.transition_valid is False:
                    stats["n_boundaries_invalid"] += 1
                    # Boundary ITSELF is collectable (we have winning_ops) —
                    # only the subsequent boundaries are dropped.
                    if bdy.winning_ops:
                        pairs.append({
                            "problem": bdy.problem,
                            "history_before": bdy.history_before,
                            "winning_ops": bdy.winning_ops,
                        })
                    break   # drop rest of trajectory
                if not bdy.winning_ops:
                    stats["n_boundaries_empty_oracle"] += 1
                    break
                pairs.append({
                    "problem": bdy.problem,
                    "history_before": bdy.history_before,
                    "winning_ops": bdy.winning_ops,
                })

        if (pi + 1) % log_every == 0:
            solved = stats["n_solved"]; total = stats["n_rollouts"]
            print(f"  rollout {pi+1}/{len(problems)}: "
                  f"solved={solved}/{total}={solved/max(total,1):.2%} "
                  f"pairs={len(pairs)}", flush=True)

    stats["drop_rate"] = (stats["n_boundaries_invalid"] /
                           max(stats["n_boundaries_total"], 1))
    stats["solved_rate"] = stats["n_solved"] / max(stats["n_rollouts"], 1)
    return pairs, stats


def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return distributed, rank, world_size, local_rank, device


def _manual_all_reduce_grads(params, world_size: int):
    """Sum-then-average gradients across ranks for each trainable param."""
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def _merge_stats(stat_list):
    """Merge rollout stats from all ranks into a single summary."""
    merged = {"n_rollouts": 0, "n_solved": 0,
              "n_boundaries_total": 0, "n_boundaries_invalid": 0,
              "n_boundaries_empty_oracle": 0,
              "stopped_reason": {}, "dropped_at_step": {}}
    for s in stat_list:
        for k in ("n_rollouts", "n_solved", "n_boundaries_total",
                  "n_boundaries_invalid", "n_boundaries_empty_oracle"):
            merged[k] += s.get(k, 0)
        for k, v in s.get("stopped_reason", {}).items():
            merged["stopped_reason"][k] = merged["stopped_reason"].get(k, 0) + v
    merged["drop_rate"] = merged["n_boundaries_invalid"] / max(merged["n_boundaries_total"], 1)
    merged["solved_rate"] = merged["n_solved"] / max(merged["n_rollouts"], 1)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_dagger.yaml")
    parser.add_argument("--use_z", action="store_true",
                        help="Inject z at step boundaries. Omit for the no-z control arm.")
    parser.add_argument("--arm_tag", default=None,
                        help="Optional override for output subdir suffix.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, rank, world_size, local_rank, device = setup_distributed()

    arm_tag = args.arm_tag or ("z" if args.use_z else "noz")
    out_root = Path(config["training"]["output_dir"]) / arm_tag
    results_root = Path(config["eval"]["output_dir"]) / arm_tag
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
        results_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "config.yaml", "w") as f:
            yaml.dump({**config, "use_z": args.use_z, "arm_tag": arm_tag,
                       "world_size": world_size}, f)

    # Deterministic seed before LoRA + up_proj init so every rank starts identical.
    seed = int(config["training"].get("seed", 1234))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tokenizer, model, head, up_proj = load_base_and_head(
        base_path=config["model"]["base_model"],
        head_path=config["model"]["head_checkpoint"],
        up_proj_hidden=int(config["model"]["up_proj_hidden"]),
        lora_r=int(config["model"]["lora_r"]),
        lora_alpha=int(config["model"]["lora_alpha"]),
        lora_dropout=float(config["model"]["lora_dropout"]),
        device=device,
    )
    if distributed:
        dist.barrier()

    all_problems = load_problems(config["data"]["train_data"])
    if int(config["data"].get("train_limit", -1)) > 0:
        all_problems = all_problems[: int(config["data"]["train_limit"])]
    # Shard problems across ranks for the rollout phase.
    local_problems = all_problems[rank::world_size]
    if rank == 0:
        print(f"arm={arm_tag} use_z={args.use_z} world_size={world_size} "
              f"n_train_problems_total={len(all_problems)} "
              f"local_shard={len(local_problems)}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += [p for p in up_proj.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable params: lora={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M "
              f"up_proj={sum(p.numel() for p in up_proj.parameters() if p.requires_grad)/1e6:.2f}M",
              flush=True)

    lr = float(config["training"]["lr"])
    wd = float(config["training"].get("weight_decay", 0.0))
    epochs = int(config["training"]["epochs"])
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd)
    grad_clip = float(config["training"].get("grad_clip", 1.0))

    log_path = out_root / "train.jsonl"

    global_step = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        if rank == 0:
            print(f"\n=== epoch {epoch} [arm={arm_tag}] ===", flush=True)

        # ---- Rollout phase (sharded; each rank rollouts its local_problems) ----
        if rank == 0:
            print("rollout phase...", flush=True)
        rollout_start = time.time()
        local_pairs, local_stats = collect_training_pairs(
            model, tokenizer, head, up_proj, local_problems,
            use_z=args.use_z, device=device,
            temperature=float(config["training"]["rollout_temperature"]),
            top_p=float(config["training"]["rollout_top_p"]),
            rollouts_per_problem=int(config["training"]["rollouts_per_problem"]),
            max_new_tokens=int(config["training"]["rollout_max_new_tokens"]),
            max_steps=int(config["training"]["max_steps"]),
            log_every=max(50 // max(world_size, 1), 25),
        )

        # Gather pairs from all ranks → every rank has the full pair list so
        # we can re-shard for training. Pairs are small Python dicts with
        # Fractions and tuples — pickleable.
        if distributed:
            all_pairs_gathered = [None] * world_size
            all_stats_gathered = [None] * world_size
            dist.all_gather_object(all_pairs_gathered, local_pairs)
            dist.all_gather_object(all_stats_gathered, local_stats)
            pairs = [p for chunk in all_pairs_gathered for p in chunk]
            r_stats = _merge_stats(all_stats_gathered)
        else:
            pairs = local_pairs
            r_stats = local_stats

        rollout_elapsed = time.time() - rollout_start
        if rank == 0:
            print(f"  rollout done in {rollout_elapsed:.0f}s | "
                  f"solved={r_stats['solved_rate']:.2%} "
                  f"drop_rate={r_stats['drop_rate']:.2%} "
                  f"n_pairs={len(pairs)}", flush=True)
            print(f"  stopped_reason: {r_stats['stopped_reason']}", flush=True)
            if epoch > 0 and r_stats['drop_rate'] > 0.50:
                print(f"  WARNING: drop_rate > 50% after warm start — "
                      f"check rollout / tokenization / oracle agreement.",
                      flush=True)

        # ---- Training phase ----
        # Deterministic shuffle (same on all ranks), then rank R takes pairs[R::world_size]
        # so every rank does approximately the same number of forward/backward passes.
        # Truncate to min-across-ranks so lockstep all_reduce can't deadlock.
        shuf_rng = random.Random(seed * 1000 + epoch)
        shuf_rng.shuffle(pairs)
        local_train_pairs = pairs[rank::world_size]
        if distributed:
            local_len = torch.tensor([len(local_train_pairs)], device=device)
            dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
            min_len = int(local_len.item())
            local_train_pairs = local_train_pairs[:min_len]
        if rank == 0:
            print(f"training phase... (local_train_pairs={len(local_train_pairs)})",
                  flush=True)
        train_start = time.time()
        model.train()
        accum_loss = 0.0; n_loss = 0
        for pi, pair in enumerate(local_train_pairs):
            optimizer.zero_grad()
            loss = dagger_loss(model, tokenizer, head, up_proj,
                               problem=pair["problem"],
                               history=pair["history_before"],
                               winning_ops=pair["winning_ops"],
                               use_z=args.use_z, device=device)
            if loss is None:
                # Still all_reduce nothing — need to keep ranks in lockstep.
                if distributed:
                    # Create a dummy loss that contributes zero grad but ensures
                    # collectives fire. Simplest: just skip this iteration and
                    # advance global_step uniformly.
                    pass
                continue
            loss.backward()
            if distributed:
                _manual_all_reduce_grads(trainable_params, world_size)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            accum_loss += float(loss.item()); n_loss += 1
            global_step += 1

            if rank == 0 and global_step % 50 == 0:
                avg = accum_loss / max(n_loss, 1)
                print(f"  step {global_step} (epoch {epoch} pair {pi+1}/{len(local_train_pairs)}) "
                      f"loss={avg:.4f}", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "epoch": epoch,
                        "pair_idx": pi+1, "pair_total": len(local_train_pairs),
                        "loss": round(avg, 4),
                    }) + "\n")
                accum_loss = 0.0; n_loss = 0

        if distributed:
            dist.barrier()
        train_elapsed = time.time() - train_start
        if rank == 0:
            print(f"  train done in {train_elapsed:.0f}s", flush=True)

        # ---- Save checkpoint + epoch stats (rank 0 only) ----
        if rank == 0:
            model.save_pretrained(out_root / f"lora_epoch{epoch}")
            torch.save(up_proj.state_dict(), out_root / f"up_projector_epoch{epoch}.pt")
            with open(out_root / f"rollout_stats_epoch{epoch}.json", "w") as f:
                json.dump(r_stats, f, indent=2)
            epoch_elapsed = time.time() - epoch_start
            print(f"=== epoch {epoch} complete in {epoch_elapsed:.0f}s ===", flush=True)
        if distributed:
            dist.barrier()

    if rank == 0:
        model.save_pretrained(out_root / "lora")
        torch.save(up_proj.state_dict(), out_root / "up_projector.pt")
        print(f"Final artifacts saved to {out_root}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
