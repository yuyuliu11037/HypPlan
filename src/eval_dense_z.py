"""Dense state-aware z injection eval for OOD probes (G24, PQ, BW, GC).

Two modes for injecting z during generation:
- `single`: compute z once from initial state, inject once at start of
  generation, then greedy-decode with no further injection. (This is what
  `src/eval_ood_generic.py` did for OOD tasks.)
- `dense`: as `single`, plus detect task-specific step boundaries during
  generation, parse the just-emitted step, update the state, recompute
  `z = up_proj(head(frozen_base(render_state(updated_state))))`, and inject
  it again before the next token. (This matches DAgger training.)

Per-task plumbing:
- `g24`:  state = remaining numbers; boundary = `\\nStep N:` (or `Answer:`
  on the last step).
- `gc`:   state = list of (vertex, color); boundary = end of one line
  matching `V<i>\\s*=\\s*<color>`.
- `bw`:   state = frozenset of facts; boundary = end of one NL action line
  (each action ends with newline).
- `pq`:   state = frozenset of (predicate, bool); boundary = end of one
  derivation step `Step N: since A, B` line. (Skipped by default — PQ
  outputs are short.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.head import HyperbolicHead, UpProjector


# --------------------------- Per-task plumbing ---------------------------


def build_g24_state(rec):
    """Initial state for G24: tuple of remaining numbers as Fractions."""
    return tuple(sorted(Fraction(int(n)) for n in rec["pool"]))


def render_g24_state(rec, state):
    nums = " ".join(str(int(x)) for x in state)
    return f"Numbers: {nums} | Target: {rec['target']}"


_G24_STEP_RE = re.compile(
    r"Step\s+\d+:\s+(\S+)\s+([+\-*/])\s+(\S+)\s+=\s+(\S+)")
_G24_BOUNDARY_RE = re.compile(r"\nStep \d+:")
_G24_ANSWER_RE = re.compile(r"Answer\s*:")


def g24_apply(state, step_text):
    """Parse the just-completed step and apply to state."""
    m = _G24_STEP_RE.search(step_text)
    if not m:
        return state, False
    try:
        a = Fraction(m.group(1)); b = Fraction(m.group(3))
        r = Fraction(m.group(4).rstrip("."))
    except (ValueError, ZeroDivisionError):
        return state, False
    rem = list(state)
    if a not in rem or b not in rem:
        return state, False
    rem.remove(a); rem.remove(b)
    rem.append(r)
    return tuple(sorted(rem)), True


def g24_boundary(full_gen, prev_count):
    """Returns (new_count, just_completed_substr) or (prev_count, None)."""
    cur = len(_G24_BOUNDARY_RE.findall(full_gen))
    if cur > prev_count:
        return cur, full_gen
    if _G24_ANSWER_RE.search(full_gen) and cur <= prev_count:
        return prev_count + 1, full_gen
    return prev_count, None


def build_gc_state(rec):
    """Initial state for GC: empty tuple of (v, color)."""
    return tuple()


def render_gc_state(rec, state):
    from src.oracle_graphcolor import Problem, render_state
    p = Problem(n=rec["n"], edges=tuple(map(tuple, rec["edges"])))
    return render_state(p, state)


_GC_LINE_RE = re.compile(r"V(\d+)\s*=\s*(\w+)")


def gc_apply(state, step_text):
    """Parse the latest 'V<i> = color' line and add to state."""
    m = _GC_LINE_RE.search(step_text)
    if not m:
        return state, False
    v = int(m.group(1))
    name = m.group(2).lower()
    code = {"red": "R", "green": "G", "blue": "B",
             "r": "R", "g": "G", "b": "B"}.get(name)
    if code is None:
        return state, False
    if any(vv == v for vv, _ in state):
        return state, False
    return tuple(sorted(state + ((v, code),), key=lambda x: x[0])), True


def gc_boundary(full_gen, prev_count):
    """Boundary = end of an assignment line. Count matches over the WHOLE
    text; when count grows, a new assignment was just emitted."""
    cur = len(_GC_LINE_RE.findall(full_gen))
    if cur > prev_count:
        return cur, full_gen
    return prev_count, None


def build_bw_state(rec):
    from src.oracle_blocksworld import parse_problem
    return parse_problem(rec["prompt"]).init


def render_bw_state(rec, state):
    from src.oracle_blocksworld import parse_problem, render_state
    p = parse_problem(rec["prompt"])
    return render_state(p, state)


_BW_NL_ACTIONS = [
    (re.compile(r"unstack the (\w+) block from on top of the (\w+) block",
                 re.I), "unstack"),
    (re.compile(r"stack the (\w+) block on top of the (\w+) block", re.I),
     "stack"),
    (re.compile(r"pick(?:[ -])up the (\w+) block", re.I), "pick-up"),
    (re.compile(r"put down the (\w+) block", re.I), "put-down"),
]


def bw_apply(state, step_text):
    from src.oracle_blocksworld import Action, apply_action
    line = step_text.split("\n")[-1].strip() or step_text.split("\n")[-2].strip() if "\n" in step_text else step_text
    # Match the LAST line in step_text
    last_lines = [ln.strip() for ln in step_text.split("\n") if ln.strip()]
    if not last_lines:
        return state, False
    line = last_lines[-1]
    for pat, op in _BW_NL_ACTIONS:
        m = pat.search(line)
        if m:
            args = tuple(g.lower() for g in m.groups())
            a = Action(op=op, args=args)
            new = apply_action(state, a)
            if new == state:
                return state, False
            return new, True
    return state, False


_BW_NEWLINE_RE = re.compile(r"\n")


def bw_boundary(full_gen, prev_count):
    """Boundary = newline (each action is on its own line)."""
    cur = len(_BW_NEWLINE_RE.findall(full_gen))
    if cur > prev_count:
        return cur, full_gen
    return prev_count, None


# Task registry
TASKS: dict[str, dict] = {
    "g24": {
        "build_state": build_g24_state,
        "render_state": render_g24_state,
        "apply": g24_apply,
        "boundary": g24_boundary,
        "max_steps": 4,   # cap z re-injections to avoid runaway
    },
    "gc": {
        "build_state": build_gc_state,
        "render_state": render_gc_state,
        "apply": gc_apply,
        "boundary": gc_boundary,
        "max_steps": 6,
    },
    "bw": {
        "build_state": build_bw_state,
        "render_state": render_bw_state,
        "apply": bw_apply,
        "boundary": bw_boundary,
        "max_steps": 12,
    },
}


# --------------------------- Inference loop ---------------------------


@torch.no_grad()
def run_one(model, tokenizer, head, up_proj, prompt: str, rec: dict,
             task_cfg: dict, max_new_tokens: int, mode: str, device) -> str:
    build_state = task_cfg["build_state"]
    render_state = task_cfg["render_state"]
    apply_fn = task_cfg["apply"]
    boundary_fn = task_cfg["boundary"]
    max_steps = task_cfg["max_steps"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    # Compute initial z and inject once.
    state = build_state(rec)
    state_text = render_state(rec, state)
    with model.disable_adapter():
        ids = tokenizer.encode(state_text, return_tensors="pt").to(device)
        sout = model(input_ids=ids, output_hidden_states=True)
        last_h = sout.hidden_states[-1][:, -1, :]
    z = up_proj(head(last_h.float())).to(torch.bfloat16).unsqueeze(1)
    out2 = model(inputs_embeds=z, past_key_values=past, use_cache=True)
    past = out2.past_key_values
    logits = out2.logits[:, -1, :]

    generated_ids: list[int] = []
    prev_boundary_count = 0
    n_injections = 1   # we already injected once

    for step in range(max_new_tokens):
        next_tok = int(logits.argmax(dim=-1).item())
        if next_tok == tokenizer.eos_token_id:
            break
        generated_ids.append(next_tok)
        cur_in = torch.tensor([[next_tok]], device=device)
        out3 = model(input_ids=cur_in, past_key_values=past, use_cache=True)
        past = out3.past_key_values
        logits = out3.logits[:, -1, :]

        if mode != "dense":
            continue
        if n_injections > max_steps:
            continue
        full_gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_count, ctx = boundary_fn(full_gen, prev_boundary_count)
        if new_count > prev_boundary_count and ctx is not None:
            prev_boundary_count = new_count
            new_state, ok = apply_fn(state, ctx)
            if ok:
                state = new_state
                state_text = render_state(rec, state)
                with model.disable_adapter():
                    ids = tokenizer.encode(state_text, return_tensors="pt").to(device)
                    sout = model(input_ids=ids, output_hidden_states=True)
                    last_h = sout.hidden_states[-1][:, -1, :]
                z = up_proj(head(last_h.float())).to(torch.bfloat16).unsqueeze(1)
                out4 = model(inputs_embeds=z, past_key_values=past, use_cache=True)
                past = out4.past_key_values
                logits = out4.logits[:, -1, :]
                n_injections += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def build_prompt(task: str, rec: dict, tokenizer) -> str:
    if task == "g24":
        # Use the same fewshot prompt as the rest of our G24 evals.
        from src.prompt_builders import fewshot_chat_prompt_24
        chat, _ = fewshot_chat_prompt_24(tokenizer, rec["problem"])
        return chat
    if task in ("pq", "bw", "gc"):
        # These already carry full prompts in `rec["prompt"]`; wrap once
        # more in the chat template.
        msgs = [{"role": "user", "content": rec["prompt"]}]
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["g24", "pq", "bw", "gc"])
    ap.add_argument("--mode", required=True, choices=["single", "dense"])
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--head_path", default=None,
                     help="Override the head; defaults to ckpt_dir's config.")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Loading {args.base_model}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"Attaching LoRA {args.ckpt_dir}/lora", flush=True)
    model = PeftModel.from_pretrained(base, str(Path(args.ckpt_dir) / "lora"))
    model.eval()

    with open(Path(args.ckpt_dir) / "config.yaml") as f:
        ckpt_cfg = yaml.safe_load(f)
    head_path = args.head_path or ckpt_cfg["model"]["head_checkpoint"]
    print(f"Loading head {head_path}", flush=True)
    sd = torch.load(head_path, map_location=device, weights_only=False)
    in_dim = sd["in_dim"]
    mc = sd["config"]["model"]
    head = HyperbolicHead(in_dim=in_dim, hyp_dim=mc["hyp_dim"],
                           hidden_dims=mc["head_hidden_dims"],
                           manifold=mc["manifold"]).to(device).float()
    head.load_state_dict(sd["state_dict"]); head.eval()
    for p in head.parameters(): p.requires_grad = False
    up_in = mc["hyp_dim"] + (1 if mc["manifold"] == "lorentz" else 0)
    up_proj = UpProjector(in_dim=up_in,
                           hidden=int(ckpt_cfg["model"]["up_proj_hidden"]),
                           out_dim=base.config.hidden_size).to(device).float()
    up_proj.load_state_dict(torch.load(
        Path(args.ckpt_dir) / "up_projector.pt", map_location=device,
        weights_only=False))
    up_proj.eval()

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval task={args.task} mode={args.mode} on {len(records)} records "
           f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    task_cfg = TASKS[args.task]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            prompt = build_prompt(args.task, rec, tok)
            gen = run_one(model, tok, head, up_proj, prompt, rec, task_cfg,
                            args.max_new_tokens, args.mode, device)
            fout.write(json.dumps({**rec, "generation": gen,
                                     "mode": args.mode, "task": args.task}) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                       f"({rate:.2f}/s)", flush=True)
    print(f"  [r{args.shard_rank}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
