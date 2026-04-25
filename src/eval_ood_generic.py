"""OOD eval driver for ProntoQA + Blocksworld.

Modes:
- `--mode base`: use the base model, no LoRA, no z. Fewshot prompt only.
- `--mode lora`: load LoRA adapter, no z injection.
- `--mode lora_randz`: load LoRA, inject one random Gaussian z (norm-matched
  to the base's hidden std) at the start of the assistant turn.

For OOD tasks we have NO task-specific Stage-1 head, so we cannot compute a
meaningful z. The `lora_randz` arm is a noise control: does the LoRA's
"z-handling" behavior survive on radically different reasoning types?
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_prompt_chat(tokenizer, prompt: str) -> tuple[str, bool]:
    """Wrap raw prompt in Qwen chat template."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    return text, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                     choices=["base", "lora", "lora_randz"])
    ap.add_argument("--ckpt_dir", default=None,
                     help="Required for lora and lora_randz modes; the LoRA "
                          "stage-2 checkpoint dir (e.g. dagger_stage2_24_varied_bal_r4/z_s1234).")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--rand_z_seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    ).to(device)

    if args.mode in {"lora", "lora_randz"}:
        if args.ckpt_dir is None:
            raise ValueError("--ckpt_dir required for lora modes")
        print(f"Attaching LoRA {args.ckpt_dir}/lora", flush=True)
        model = PeftModel.from_pretrained(base, str(Path(args.ckpt_dir) / "lora"))
        model.eval()
    else:
        model = base
        model.eval()

    # For lora_randz, we'll inject one random Gaussian as virtual token.
    # Norm: sqrt(hidden_dim) — same scale as the trained up_projector's
    # LayerNorm-normalized output.
    hidden_dim = base.config.hidden_size

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"Eval mode={args.mode} on {len(records)} records "
          f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    g = torch.Generator(device=device).manual_seed(args.rand_z_seed)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_done = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for rec in records:
            prompt_text, add_special = _build_prompt_chat(tokenizer,
                                                            rec["prompt"])
            input_ids = tokenizer.encode(prompt_text,
                                          add_special_tokens=add_special,
                                          return_tensors="pt").to(device)

            with torch.no_grad():
                if args.mode == "lora_randz":
                    # Forward the prompt to get past_kv, then inject random z
                    # as one virtual token, then do a single autoregressive
                    # generate using past_kv.
                    out = model(input_ids=input_ids, use_cache=True)
                    past = out.past_key_values
                    z = torch.randn(1, 1, hidden_dim, device=device,
                                     generator=g).to(torch.bfloat16)
                    z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    z = z * (hidden_dim ** 0.5)
                    out2 = model(inputs_embeds=z, past_key_values=past,
                                  use_cache=True)
                    past = out2.past_key_values
                    next_tok = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = [int(next_tok.item())]
                    cur = next_tok
                    for _ in range(args.max_new_tokens - 1):
                        if int(cur.item()) == tokenizer.eos_token_id:
                            break
                        out3 = model(input_ids=cur, past_key_values=past,
                                      use_cache=True)
                        past = out3.past_key_values
                        cur = out3.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated.append(int(cur.item()))
                    gen = tokenizer.decode(generated, skip_special_tokens=True)
                else:
                    out_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(args.temperature > 0),
                        temperature=args.temperature if args.temperature > 0 else 1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen = tokenizer.decode(out_ids[0, input_ids.size(1):],
                                            skip_special_tokens=True)

            fout.write(json.dumps({
                **rec, "generation": gen, "mode": args.mode,
            }) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  [r{args.shard_rank}] {n_done}/{len(records)} "
                       f"({rate:.2f}/s)", flush=True)

    print(f"[r{args.shard_rank}] done in {time.time()-t0:.0f}s "
          f"→ {args.output}", flush=True)


if __name__ == "__main__":
    main()
