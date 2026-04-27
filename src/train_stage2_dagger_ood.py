"""Generic Stage-2 DAgger trainer for OOD tasks (PQ / BW / GC).

Mirrors src/train_stage2_dagger_varied.py:
  - per-epoch rollout phase (collect (state, gold_step) tuples)
  - training phase (CE on gold step text)
  - LoRA + UpProjector trainable, head + base frozen
  - DDP across multiple GPUs (gloo backend per HYPPLAN_DIST_BACKEND)

Task selection via --task {pq,bw,gc} loads the corresponding adapter +
training data file.

Training data format: {"question": ..., "answer": ..., "id": ...} with the
same id used to look up the structured Problem from the task adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dagger_ood_adapters import ADAPTERS
from src.dagger_rollout_ood import rollout_one
from src.head import HyperbolicHead, UpProjector


def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        backend = os.environ.get("HYPPLAN_DIST_BACKEND", "gloo")
        dist.init_process_group(backend=backend,
                                  timeout=timedelta(hours=8))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return distributed, rank, world_size, local_rank, device


def _manual_all_reduce_grads(params, world_size):
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def load_records(jsonl_path: str) -> list[dict]:
    """Load training records. Each must be a dict consumable by an adapter."""
    return [json.loads(l) for l in open(jsonl_path)]


def load_test_records(task: str, jsonl_path: str) -> list[dict]:
    """Load test records (slightly different schema for some tasks)."""
    return [json.loads(l) for l in open(jsonl_path)]


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


def _compute_z_with_grad(model, tokenizer, head, up_proj, adapter, state,
                          history, device, random_z: bool = False
                          ) -> torch.Tensor:
    hidden_dim = up_proj.net[-1].normalized_shape[0]
    if random_z:
        g = torch.randn(1, hidden_dim, device=device)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return g * (hidden_dim ** 0.5)
    state_text = adapter.render_state(state, history)
    ids = tokenizer.encode(state_text, add_special_tokens=True,
                            return_tensors="pt").to(device)
    with torch.no_grad():
        with model.disable_adapter():
            out = model(input_ids=ids, output_hidden_states=True)
            last_h = out.hidden_states[-1][:, -1, :]
        z_hyp = head(last_h.float())
    return up_proj(z_hyp)


def _pick_winner(winners):
    """Pick one winner deterministically. Sort by string representation of
    action for stability."""
    if not winners:
        return None
    return min(winners, key=lambda w: repr(w[0]))


def dagger_loss(model, tokenizer, head, up_proj, adapter, state_before,
                history, winners, use_z: bool, device: torch.device,
                max_steps: int = 12, random_z: bool = False
                ) -> torch.Tensor:
    """CE loss on gold step text. Reconstructs the input embedding sequence:
    prompt + Step1: + (z + step1_text + Step2: priming) + ... + (z + gold_step).

    The label mask is -100 over context, then the gold_step token ids."""
    winner = _pick_winner(winners)
    if winner is None:
        return None
    embed_table = model.get_input_embeddings()
    dtype = next(model.parameters()).dtype

    def _embed_text(text: str, add_special: bool):
        ids = tokenizer.encode(text, add_special_tokens=add_special,
                                return_tensors="pt").to(device)
        return embed_table(ids), ids.size(1)

    pieces = []
    prompt_text, prompt_add_special = adapter.make_prompt(tokenizer)
    prompt_text = prompt_text + adapter.step_priming_prefix(1)
    ctx_emb, _ = _embed_text(prompt_text, add_special=prompt_add_special)
    pieces.append(ctx_emb)

    # Replay each completed step in history with z prepended.
    cur_state = adapter.initial_state
    for i, action in enumerate(history):
        if use_z:
            z = _compute_z_with_grad(model, tokenizer, head, up_proj, adapter,
                                       cur_state, history[:i], device,
                                       random_z=random_z)
            pieces.append(z.unsqueeze(1).to(dtype))
        ok, ns = adapter.validate_apply(cur_state, action)
        cur_state = ns
        step_text = adapter.format_step_text(
            adapter.initial_state if i == 0 else cur_state, action, ns,
            i + 1, max_steps)
        emb, _ = _embed_text(step_text, add_special=False)
        pieces.append(emb)

    if use_z:
        z = _compute_z_with_grad(model, tokenizer, head, up_proj, adapter,
                                   state_before, history, device,
                                   random_z=random_z)
        pieces.append(z.unsqueeze(1).to(dtype))

    context = torch.cat(pieces, dim=1)
    context_len = context.size(1)

    action, ns = winner
    target_text = adapter.format_step_text(state_before, winner, ns,
                                              len(history) + 1, max_steps)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False,
                                    return_tensors="pt").to(device)
    target_embeds = embed_table(target_ids)

    combined = torch.cat([context, target_embeds], dim=1)
    pad_ids = torch.full((1, context_len), -100, dtype=torch.long,
                          device=device)
    labels = torch.cat([pad_ids, target_ids], dim=1)

    out = model(inputs_embeds=combined, labels=labels, use_cache=False)
    return out.loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_z", type=int, default=1)
    ap.add_argument("--random_z", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    distributed, rank, world_size, local_rank, device = setup_distributed()
    seed = int(config["training"].get("seed", 1234))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    AdapterCls = ADAPTERS[args.task]
    train_records = load_records(config["data"]["train_data"])
    if config["data"].get("train_limit", -1) > 0:
        train_records = train_records[: int(config["data"]["train_limit"])]
    if rank == 0:
        print(f"task={args.task}  records={len(train_records)}  "
                f"world={world_size}", flush=True)

    tokenizer, model, head, up_proj = load_base_and_head(
        base_path=config["model"]["base_model"],
        head_path=config["model"]["head_checkpoint"],
        up_proj_hidden=int(config["model"]["up_proj_hidden"]),
        lora_r=int(config["model"]["lora_r"]),
        lora_alpha=int(config["model"]["lora_alpha"]),
        lora_dropout=float(config["model"]["lora_dropout"]),
        device=device,
    )
    if rank == 0:
        model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += [p for p in up_proj.parameters() if p.requires_grad]
    optimizer = AdamW(trainable,
                       lr=float(config["training"]["lr"]),
                       weight_decay=float(
                           config["training"].get("weight_decay", 0.0)))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    epochs = int(config["training"]["epochs"])
    max_steps = int(config["training"].get("max_steps_per_problem", 12))
    rollouts_per_problem = int(
        config["training"].get("rollouts_per_problem", 1))

    out_dir = Path(config["training"]["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "config.yaml").open("w") as f:
            yaml.dump(config, f)

    log_path = out_dir / "train.jsonl"
    use_z = bool(args.use_z)
    random_z = bool(args.random_z)

    for epoch in range(epochs):
        if rank == 0:
            print(f"=== epoch {epoch} ({'rollout' if epoch >= 0 else ''}) ===",
                    flush=True)

        # ---- Rollout phase: each rank rolls out a shard of training problems ----
        my_records = train_records[rank::world_size]
        rollout_pairs = []  # (rec, state, history, winning_steps)
        model.eval()
        for ri, rec in enumerate(my_records):
            try:
                adapter = AdapterCls(rec)
            except Exception as e:
                if rank == 0 and ri < 3:
                    print(f"  [r{rank}] adapter fail rec {ri}: {e}",
                            flush=True)
                continue
            for ro in range(rollouts_per_problem):
                try:
                    r = rollout_one(
                        model, tokenizer, head, up_proj, adapter, device,
                        use_z=use_z,
                        temperature=float(
                            config["training"]["rollout_temperature"]),
                        top_p=float(config["training"]["rollout_top_p"]),
                        max_new_tokens=int(
                            config["training"]["rollout_max_new_tokens"]),
                        max_steps=max_steps,
                        random_z=random_z,
                    )
                except Exception as e:
                    if rank == 0 and ri < 3:
                        print(f"  [r{rank}] rollout fail rec {ri}: "
                                f"{type(e).__name__} {e}", flush=True)
                    continue
                for bdy in r.boundaries:
                    if not bdy.winning_steps:
                        continue
                    rollout_pairs.append({
                        "rec": rec,
                        "state": bdy.state_before,
                        "history": bdy.history_before,
                        "winners": bdy.winning_steps,
                    })
            if rank == 0 and (ri + 1) % 25 == 0:
                print(f"  [r{rank}] rollout {ri+1}/{len(my_records)} "
                        f"pairs={len(rollout_pairs)}", flush=True)

        if rank == 0:
            print(f"  [r{rank}] rollout done: {len(rollout_pairs)} (state, winner) pairs",
                    flush=True)

        # ---- Training phase ----
        # Sync the per-rank pair count to a global min so all ranks make
        # the same number of all_reduce calls (gloo blocks otherwise).
        if distributed:
            # Cyclic-pad every rank to global-MAX so all ranks call all_reduce
            # the same number of times AND no rollout data is wasted. Ranks
            # with fewer pairs repeat their own pairs (cyclic) to match the
            # busiest rank's count.
            local_n = torch.tensor([len(rollout_pairs)],
                                     dtype=torch.long, device="cpu")
            dist.all_reduce(local_n, op=dist.ReduceOp.MAX)
            global_max = int(local_n.item())
            random.shuffle(rollout_pairs)
            if len(rollout_pairs) == 0:
                # Pathological: a rank had no rollout pairs. Fill with no-ops
                # by repeating any other rank's data isn't possible without
                # cross-rank exchange — skip this epoch's training on this
                # rank by padding with None and the loop guards None loss.
                rollout_pairs = [None] * global_max
            else:
                base = rollout_pairs
                padded = []
                while len(padded) < global_max:
                    padded.extend(base)
                rollout_pairs = padded[:global_max]
            if rank == 0:
                print(f"  [sync] padding to global max pairs = "
                        f"{global_max} (rank 0 unique pre-pad)",
                        flush=True)
        else:
            random.shuffle(rollout_pairs)
        accum_loss = 0.0
        n_loss = 0
        # Initialize grads as zero tensors so every rank's all_reduce can
        # find them on iteration 1 even when this rank has no loss.
        for p in trainable:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        for pi, pair in enumerate(rollout_pairs):
            loss = None
            if pair is not None:
                try:
                    adapter = AdapterCls(pair["rec"])
                    loss = dagger_loss(model, tokenizer, head, up_proj,
                                         adapter, pair["state"],
                                         pair["history"], pair["winners"],
                                         use_z=use_z, device=device,
                                         max_steps=max_steps,
                                         random_z=random_z)
                except Exception as e:
                    if rank == 0 and pi < 3:
                        print(f"  [r{rank}] loss fail pair {pi}: "
                                f"{type(e).__name__} {e}", flush=True)
                    loss = None
            if loss is not None:
                loss.backward()
                accum_loss += float(loss.item())
                n_loss += 1
            # ALWAYS sync gradients to keep ranks aligned, even when this
            # rank had no valid loss this iteration (its grads will be
            # zeros / unchanged, contributing nothing to the average).
            if distributed:
                _manual_all_reduce_grads(trainable, world_size)
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            # Zero grads in-place (NOT to None) so the next iteration's
            # all_reduce always finds the param's grad tensor on every rank.
            optimizer.zero_grad(set_to_none=False)
            if rank == 0 and (pi + 1) % 25 == 0:
                print(f"  [r{rank}] step {pi+1}/{len(rollout_pairs)} "
                        f"avg_loss={accum_loss/max(n_loss,1):.4f}",
                        flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"epoch": epoch, "step": pi+1,
                                          "avg_loss": round(accum_loss/max(n_loss,1), 4)
                                          }) + "\n")

        if rank == 0:
            print(f"=== epoch {epoch} done; "
                    f"avg_loss={accum_loss/max(n_loss,1):.4f} ===",
                    flush=True)
            # Save end-of-epoch checkpoint so an early kill still leaves a
            # usable model. Final destination is overwritten each epoch.
            lora_dir = out_dir / "lora"
            lora_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(lora_dir)
            torch.save(up_proj.state_dict(), out_dir / "up_projector.pt")
            print(f"  [epoch {epoch}] saved LoRA + up_projector to {out_dir}",
                    flush=True)

    if rank == 0:
        print(f"Saved LoRA + up_projector to {out_dir}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
