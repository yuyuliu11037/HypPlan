"""Varied-target DAgger trainer — fork of
[src/train_stage2_dagger.py](src/train_stage2_dagger.py).

Key differences:
  - Training problems are (pool, target, n_steps) records from
    `data/24_varied_*.jsonl`, not fixed-target "a,b,c,d" strings.
  - Uses `src/dagger_rollout_varied.rollout_one` which accepts (pool, target).
  - Uses `src/oracle_24_varied.winning_ops` and `src/tree_data_generic`.
  - `_compute_z_with_grad` operates on (pool, target) and uses
    `render_state_generic`.
  - `_format_winner_target` uses the per-record `target` instead of the
    hardcoded 24 when writing "Answer: <target>".
  - Prompt builder is `fewshot_chat_prompt_generic`.

All other mechanics (manual DDP all_reduce, deterministic seeding, epoch-wise
rollout + train phases, LoRA + up_projector trainable, head + base frozen)
are unchanged. NCCL-safe GPU selection is the caller's responsibility — pair
(5, 7) is broken on this host.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dagger_rollout_varied import rollout_one
from src.head import HyperbolicHead, UpProjector
from src.prompt_builders import fewshot_chat_prompt_generic
from src.tree_data_generic import render_state_generic
from src.train_stage2_dagger import (
    _manual_all_reduce_grads,
    _merge_stats,
)


def setup_distributed():
    """Like src.train_stage2_dagger.setup_distributed but with a 3-hour NCCL
    timeout so variable-length rollouts across ranks don't cascade into a
    timeout during the all_gather_object sync."""
    from datetime import timedelta
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(
            backend="nccl", device_id=device,
            timeout=timedelta(hours=8),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return distributed, rank, world_size, local_rank, device


def load_varied_records(jsonl_path: str, skip_trivial: bool = True
                         ) -> list[dict]:
    """Load (pool, target, n_steps) records. Skips 0-step trivial ones by
    default — those have nothing for DAgger to train on."""
    out: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if skip_trivial and int(r.get("n_steps", 1)) == 0:
                continue
            out.append({
                "pool": list(r["pool"]),
                "target": int(r["target"]),
                "n_steps": int(r.get("n_steps", len(r["pool"]) - 1)),
            })
    return out


def load_base_and_head(base_path: str, head_path: str, up_proj_hidden: int,
                        lora_r: int, lora_alpha: int, lora_dropout: float,
                        device: torch.device):
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
    for m in reversed(list(up_proj.net.modules())):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            torch.nn.init.zeros_(m.bias)
            break

    return tokenizer, model, head, up_proj


def _compute_z_with_grad(model, tokenizer, head, up_proj, pool: list,
                          target: int, history: tuple, device: torch.device,
                          random_z: bool = False) -> torch.Tensor:
    hidden_dim = up_proj.net[-1].normalized_shape[0]
    if random_z:
        g = torch.randn(1, hidden_dim, device=device)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return g * (hidden_dim ** 0.5)
    state_text = render_state_generic(pool, target, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True,
                            return_tensors="pt").to(device)
    with torch.no_grad():
        with model.disable_adapter():
            out = model(input_ids=ids, output_hidden_states=True)
            last_h = out.hidden_states[-1][:, -1, :]
        z_hyp = head(last_h.float())
    return up_proj(z_hyp)


def _pick_winner(winning_ops):
    if not winning_ops:
        return None
    return min(winning_ops, key=lambda w: (w[0], float(w[1]), float(w[2])))


def _fraction_str(f: Fraction) -> str:
    return str(int(f)) if f.denominator == 1 else str(f)


def _format_winner_target(a: Fraction, op_sym: str, b: Fraction, r: Fraction,
                           pool_after: list, max_steps: int, step_num: int,
                           target: int) -> str:
    """Step text for picked winner. Varied-target variant uses per-record
    `target` in the `Answer: T` line."""
    head = f" {_fraction_str(a)} {op_sym} {_fraction_str(b)} = {_fraction_str(r)}"
    if len(pool_after) == 1 and pool_after[0] == Fraction(int(target)):
        return head + f". Answer: {int(target)}"
    rem = " ".join(_fraction_str(x) for x in sorted(pool_after))
    tail = f". Remaining: {rem}"
    next_step = step_num + 1
    if next_step <= max_steps:
        tail += f"\nStep {next_step}:"
    return head + tail


def dagger_loss(model, tokenizer, head, up_proj, pool: list, target: int,
                history: tuple, winning_ops, use_z: bool,
                device: torch.device, max_steps: int = 3,
                random_z: bool = False) -> torch.Tensor:
    winner = _pick_winner(winning_ops)
    if winner is None:
        return None
    wop, wa, wb, wr = winner

    embed_table = model.get_input_embeddings()
    dtype = next(model.parameters()).dtype

    def _embed_text(text: str, add_special: bool):
        ids = tokenizer.encode(text, add_special_tokens=add_special,
                                return_tensors="pt").to(device)
        return embed_table(ids), ids.size(1)

    pieces = []
    prompt_text, prompt_add_special = fewshot_chat_prompt_generic(
        tokenizer, pool, target)
    # Prime "Step 1:" identically to rollout so CE context matches rollout
    # context exactly. Trivial (0-step) cases are skipped upstream via
    # skip_trivial in load_varied_records.
    prompt_text = prompt_text + "Step 1:"
    ctx_prompt_embed, _ = _embed_text(prompt_text, add_special=prompt_add_special)
    pieces.append(ctx_prompt_embed)

    cur_pool = [Fraction(int(x)) for x in pool]

    for i, (a, op_sym, b, r) in enumerate(history):
        if use_z:
            z = _compute_z_with_grad(model, tokenizer, head, up_proj,
                                      pool, target, history[:i], device,
                                      random_z=random_z)
            pieces.append(z.unsqueeze(1).to(dtype))
        cur_pool.remove(a); cur_pool.remove(b); cur_pool.append(r)
        remaining_sorted = sorted(cur_pool)
        content = (
            f" {_fraction_str(a)} {op_sym} {_fraction_str(b)} = "
            f"{_fraction_str(r)}. Remaining: "
            + " ".join(_fraction_str(x) for x in remaining_sorted)
        )
        next_step_num = i + 2
        if next_step_num <= max_steps:
            content += f"\nStep {next_step_num}:"
        emb, _ = _embed_text(content, add_special=False)
        pieces.append(emb)

    if use_z:
        z = _compute_z_with_grad(model, tokenizer, head, up_proj,
                                  pool, target, history, device,
                                  random_z=random_z)
        pieces.append(z.unsqueeze(1).to(dtype))

    context = torch.cat(pieces, dim=1)
    context_len = context.size(1)

    pool_after_winner = list(cur_pool)
    pool_after_winner.remove(wa); pool_after_winner.remove(wb)
    pool_after_winner.append(wr)
    step_num = len(history) + 1
    target_text = _format_winner_target(wa, wop, wb, wr, pool_after_winner,
                                         max_steps, step_num, target)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False,
                                    return_tensors="pt").to(device)
    target_embeds = embed_table(target_ids)

    combined = torch.cat([context, target_embeds], dim=1)
    pad_ids = torch.full((1, context_len), -100, dtype=torch.long,
                          device=device)
    labels = torch.cat([pad_ids, target_ids], dim=1)

    out = model(inputs_embeds=combined, labels=labels, use_cache=False)
    return out.loss


def collect_training_pairs(model, tokenizer, head, up_proj, records,
                             use_z: bool, device, temperature: float,
                             top_p: float, rollouts_per_problem: int,
                             max_new_tokens: int, max_steps: int,
                             log_every: int = 100,
                             random_z: bool = False):
    model.eval()
    pairs = []
    stats = {"n_rollouts": 0, "n_solved": 0,
             "stopped_reason": {}, "dropped_at_step": {},
             "n_boundaries_total": 0, "n_boundaries_invalid": 0,
             "n_boundaries_empty_oracle": 0}

    for pi, rec in enumerate(records):
        pool = rec["pool"]; target = rec["target"]
        for ri in range(rollouts_per_problem):
            r = rollout_one(model, tokenizer, head, up_proj, pool, target,
                             device, use_z=use_z, temperature=temperature,
                             top_p=top_p, max_new_tokens=max_new_tokens,
                             max_steps=max_steps, random_z=random_z)
            stats["n_rollouts"] += 1
            stats["n_solved"] += int(r.solved)
            stats["stopped_reason"][r.stopped_reason] = (
                stats["stopped_reason"].get(r.stopped_reason, 0) + 1)

            for bdy in r.boundaries:
                stats["n_boundaries_total"] += 1
                if bdy.transition_valid is False:
                    stats["n_boundaries_invalid"] += 1
                    if bdy.winning_ops:
                        pairs.append({
                            "pool": bdy.pool, "target": bdy.target,
                            "history_before": bdy.history_before,
                            "winning_ops": bdy.winning_ops,
                        })
                    break
                if not bdy.winning_ops:
                    stats["n_boundaries_empty_oracle"] += 1
                    break
                pairs.append({
                    "pool": bdy.pool, "target": bdy.target,
                    "history_before": bdy.history_before,
                    "winning_ops": bdy.winning_ops,
                })

        if (pi + 1) % log_every == 0:
            solved = stats["n_solved"]; total = stats["n_rollouts"]
            print(f"  rollout {pi+1}/{len(records)}: "
                  f"solved={solved}/{total}={solved/max(total,1):.2%} "
                  f"pairs={len(pairs)}", flush=True)

    stats["drop_rate"] = (stats["n_boundaries_invalid"] /
                           max(stats["n_boundaries_total"], 1))
    stats["solved_rate"] = stats["n_solved"] / max(stats["n_rollouts"], 1)
    return pairs, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--use_z", action="store_true")
    parser.add_argument("--random_z", action="store_true")
    parser.add_argument("--arm_tag", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, rank, world_size, local_rank, device = setup_distributed()

    if args.random_z and not args.use_z:
        raise SystemExit("--random_z requires --use_z")

    if args.seed is not None:
        config.setdefault("training", {})["seed"] = int(args.seed)
        base_arm = ("randz" if args.random_z else
                     ("z" if args.use_z else "noz"))
        default_arm = f"{base_arm}_s{int(args.seed)}"
    else:
        default_arm = ("randz" if args.random_z else
                        ("z" if args.use_z else "noz"))
    arm_tag = args.arm_tag or default_arm
    out_root = Path(config["training"]["output_dir"]) / arm_tag
    results_root = Path(config["eval"]["output_dir"]) / arm_tag
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
        results_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "config.yaml", "w") as f:
            yaml.dump({**config, "use_z": args.use_z,
                       "random_z": args.random_z,
                       "arm_tag": arm_tag, "world_size": world_size}, f)

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

    all_records = load_varied_records(config["data"]["train_data"],
                                        skip_trivial=True)
    if int(config["data"].get("train_limit", -1)) > 0:
        all_records = all_records[: int(config["data"]["train_limit"])]
    local_records = all_records[rank::world_size]
    if rank == 0:
        print(f"arm={arm_tag} use_z={args.use_z} random_z={args.random_z} "
              f"world_size={world_size} "
              f"n_train_records_total={len(all_records)} "
              f"local_shard={len(local_records)}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += [p for p in up_proj.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable params: "
              f"lora={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M "
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

        if rank == 0:
            print("rollout phase...", flush=True)
        rollout_start = time.time()
        local_pairs, local_stats = collect_training_pairs(
            model, tokenizer, head, up_proj, local_records,
            use_z=args.use_z, device=device,
            temperature=float(config["training"]["rollout_temperature"]),
            top_p=float(config["training"]["rollout_top_p"]),
            rollouts_per_problem=int(config["training"]["rollouts_per_problem"]),
            max_new_tokens=int(config["training"]["rollout_max_new_tokens"]),
            max_steps=int(config["training"].get("max_steps", 3)),
            log_every=max(50 // max(world_size, 1), 25),
            random_z=args.random_z,
        )

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

        # Training phase
        shuf_rng = random.Random(seed * 1000 + epoch)
        shuf_rng.shuffle(pairs)
        local_train_pairs = pairs[rank::world_size]
        if distributed:
            local_len = torch.tensor([len(local_train_pairs)], device=device)
            dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
            min_len = int(local_len.item())
            local_train_pairs = local_train_pairs[:min_len]
        if rank == 0:
            print(f"training phase... "
                  f"(local_train_pairs={len(local_train_pairs)})", flush=True)
        train_start = time.time()
        model.train()
        accum_loss = 0.0; n_loss = 0
        for pi, pair in enumerate(local_train_pairs):
            optimizer.zero_grad()
            loss = dagger_loss(
                model, tokenizer, head, up_proj,
                pool=pair["pool"], target=pair["target"],
                history=pair["history_before"],
                winning_ops=pair["winning_ops"],
                use_z=args.use_z, device=device,
                max_steps=int(config["training"].get("max_steps", 3)),
                random_z=args.random_z,
            )
            if loss is None:
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
                print(f"  step {global_step} (epoch {epoch} pair {pi+1}/"
                      f"{len(local_train_pairs)}) loss={avg:.4f}", flush=True)
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
            model.save_pretrained(out_root / f"lora_epoch{epoch}")
            torch.save(up_proj.state_dict(),
                        out_root / f"up_projector_epoch{epoch}.pt")
            with open(out_root / f"rollout_stats_epoch{epoch}.json", "w") as f:
                json.dump(r_stats, f, indent=2)
            epoch_elapsed = time.time() - epoch_start
            print(f"=== epoch {epoch} complete in {epoch_elapsed:.0f}s ===",
                  flush=True)
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
