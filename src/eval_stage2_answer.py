"""Answer-accuracy eval for HypPlan in-domain Stage-2 LoRA + per-task head.

Standard final-answer-accuracy protocol for tasks where the gold
trajectory may have zero rule-application steps (ProofWriter QDep=0,
CWA-False, CWA-double-negative). Distinct from
src/eval_stage2_indomain.py, which uses strict step-by-step oracle
gating and bails when winning_steps is empty.

Pipeline per record:
  1. Build prompt via adapter.make_prompt
  2. Compute initial-state z using head + up_projector (HypPlan's
     virtual-token conditioning)
  3. Inject z as a virtual token between prompt and generation
  4. Generate freely up to max_new_tokens (no oracle gating)
  5. Parse final answer from the generated text
  6. Score against gold

Reports overall accuracy plus per-difficulty breakdown
(--breakdown_field, default "QDep" for ProofWriter).

Sharding via --shard_rank/--shard_world (records assigned modulo).

Usage:
  python3.10 -m src.eval_stage2_answer \\
    --task proofwriter \\
    --ckpt_dir checkpoints/dagger_stage2_proofwriter_indomain \\
    --head_path checkpoints/head_proofwriter_qwen14b_rank/head.pt \\
    --test_data data/proofwriter_test.jsonl \\
    --output results/eval_stage2_indomain/proofwriter/proofwriter_answer.jsonl \\
    --shard_rank 0 --shard_world 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dagger_ood_adapters import ADAPTERS
from src.head import HyperbolicHead, UpProjector


def parse_answer(task: str, gen: str, rec: dict):
    """Return a hashable representation of the model's final answer, or
    None if unparseable. Mirrors src/eval_baseline_kpath.extract_answer_key."""
    if task == "proofwriter":
        m = re.search(r"Answer\s*[:\-]?\s*(True|False)", gen,
                      re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
        return None
    if task == "pq":
        m = re.search(r"\b(A|B)\b", gen.split("\n", 1)[0])
        return m.group(1) if m else None
    if task in ("rulechain", "synthlogic"):
        m = re.search(r"Answer\s*[:\-]?\s*(.+?)\s*$", gen, re.MULTILINE)
        return m.group(1).strip() if m else None
    if task == "clutrr":
        from src.oracle_clutrr import parse_answer as pa
        return pa(gen)
    if task == "nqueens":
        from src.oracle_nqueens import parse_solution
        sol = parse_solution(gen)
        return tuple(sol) if sol else None
    raise NotImplementedError(f"parse_answer for task={task}")


def is_correct(task: str, parsed, rec: dict) -> bool:
    if parsed is None:
        return False
    if task == "proofwriter":
        return bool(parsed) == bool(rec["answer"])
    if task == "pq":
        return parsed == rec.get("answer_label")
    if task in ("rulechain", "synthlogic", "clutrr"):
        return parsed == rec.get("answer", rec.get("answer_label"))
    if task == "nqueens":
        from src.oracle_nqueens import score_solution
        N = int(rec["N"])
        prefix = list(rec.get("prefix", []))
        sol = list(parsed)
        if len(sol) != N:
            return False
        if sol[: len(prefix)] != prefix:
            return False
        return score_solution(N, sol)
    raise NotImplementedError(f"is_correct for task={task}")


@torch.no_grad()
def _compute_initial_z(model, tok, head, up_proj, adapter, device):
    state_text = adapter.render_state(adapter.initial_state, tuple())
    ids = tok.encode(state_text, add_special_tokens=True,
                     return_tensors="pt").to(device)
    with model.disable_adapter():
        out = model(input_ids=ids, output_hidden_states=True)
        last_h = out.hidden_states[-1][:, -1, :]
    z_hyp = head(last_h.float())
    return up_proj(z_hyp)


@torch.no_grad()
def generate_with_z(model, tok, prompt_text: str, z_token, device,
                    max_new_tokens: int, temperature: float, top_p: float):
    """Inject z as a virtual token between prompt and generation, then
    decode greedily (or sample) up to max_new_tokens."""
    input_ids = tok.encode(prompt_text, return_tensors="pt").to(device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values

    if z_token is not None:
        embeds = z_token.unsqueeze(1).to(next(model.parameters()).dtype)
        out = model(inputs_embeds=embeds, past_key_values=past,
                    use_cache=True)
        past = out.past_key_values
    logits = out.logits[:, -1, :]

    generated: list[int] = []
    for _ in range(max_new_tokens):
        if temperature <= 0:
            nxt = int(logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            if 0 < top_p < 1:
                sp, si = probs.sort(dim=-1, descending=True)
                cs = sp.cumsum(dim=-1)
                mask = cs - sp > top_p
                sp[mask] = 0.0
                sp /= sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                pick = torch.multinomial(sp, 1)
                nxt = int(si.gather(-1, pick).item())
            else:
                nxt = int(torch.multinomial(probs, 1).item())
        if nxt == tok.eos_token_id:
            break
        generated.append(nxt)
        cur = torch.tensor([[nxt]], device=device)
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return tok.decode(generated, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--head_path", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--use_z", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--breakdown_field", default="QDep",
                    help="record field to break down accuracy by "
                         "(default QDep for ProofWriter)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"Attaching LoRA {args.ckpt_dir}/lora", flush=True)
    model = PeftModel.from_pretrained(base, str(Path(args.ckpt_dir) / "lora"))
    model.eval()

    sd = torch.load(args.head_path, map_location=device, weights_only=False)
    in_dim = sd["in_dim"]
    mc = sd["config"]["model"]
    head = HyperbolicHead(in_dim=in_dim, hyp_dim=mc["hyp_dim"],
                          hidden_dims=mc["head_hidden_dims"],
                          manifold=mc["manifold"]).to(device).float()
    head.load_state_dict(sd["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    with open(Path(args.ckpt_dir) / "config.yaml") as f:
        ckpt_cfg = yaml.safe_load(f)
    up_in = mc["hyp_dim"] + (1 if mc["manifold"] == "lorentz" else 0)
    up_proj = UpProjector(in_dim=up_in,
                          hidden=int(ckpt_cfg["model"]["up_proj_hidden"]),
                          out_dim=base.config.hidden_size).to(device).float()
    up_proj.load_state_dict(torch.load(
        Path(args.ckpt_dir) / "up_projector.pt", map_location=device,
        weights_only=False))
    up_proj.eval()

    AdapterCls = ADAPTERS[args.task]
    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval Stage-2 answer-accuracy {args.task} on {len(records)} "
          f"records (shard {args.shard_rank}/{args.shard_world})",
          flush=True)

    out_path = Path(args.output)
    if args.shard_world > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}_shard{args.shard_rank}{out_path.suffix}"
        )
    args.output = str(out_path)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_correct = 0
    breakdown = {}
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            try:
                adapter = AdapterCls(rec)
                prompt_text, _ = adapter.make_prompt(tok)
                z_token = (_compute_initial_z(model, tok, head, up_proj,
                                              adapter, device)
                           if args.use_z else None)
                gen = generate_with_z(
                    model, tok, prompt_text, z_token, device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
            except Exception as e:
                fout.write(json.dumps({**{k: rec[k] for k in rec
                                          if k != "prompt"},
                                       "error": f"{type(e).__name__}: {e}"[:200]
                                       }) + "\n")
                fout.flush()
                continue

            parsed = parse_answer(args.task, gen, rec)
            ok = is_correct(args.task, parsed, rec)
            n_correct += int(ok)

            bd_key = rec.get(args.breakdown_field)
            if bd_key is not None:
                bk = breakdown.setdefault(bd_key, [0, 0])
                bk[0] += 1
                bk[1] += int(ok)

            fout.write(json.dumps({
                **{k: rec[k] for k in rec if k != "prompt"},
                "generation": gen,
                "parsed_answer": parsed if not isinstance(parsed, frozenset)
                else list(parsed),
                "correct": ok,
            }) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0 or i == len(records) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(records) - (i + 1)) / rate
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                      f"acc={n_correct/(i+1):.3f} "
                      f"({elapsed/60:.1f}m, eta={eta/60:.1f}m)",
                      flush=True)

    print(f"\n[r{args.shard_rank}] {n_correct}/{len(records)} = "
          f"{n_correct/len(records):.3f}", flush=True)
    if breakdown:
        print(f"breakdown by {args.breakdown_field}:", flush=True)
        for k in sorted(breakdown):
            t, c = breakdown[k]
            print(f"  {args.breakdown_field}={k}: {c}/{t} = {c/t:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
