from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.math_grading import extract_final_answer, grade_answer, has_complete_boxed
from src.model.planning_model import apply_plan_token_logit_delta
from src.model.proj import ProjectionModule


INSTRUCTION_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n"
    "Solution:\n"
)


def ddp_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def shard(items: list[dict], rank: int, world_size: int) -> list[dict]:
    return [x for i, x in enumerate(items) if i % world_size == rank]


def summarize(results: list[dict]) -> dict:
    overall = sum(int(r["correct"]) for r in results) / max(1, len(results))
    by_level = defaultdict(list)
    by_subject = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(int(r["correct"]))
        by_subject[r["subject"]].append(int(r["correct"]))
    return {
        "overall_accuracy": overall,
        "count": len(results),
        "avg_generated_tokens": sum(r["generated_tokens"] for r in results) / max(1, len(results)),
        "avg_plan_tokens_emitted": sum(r["plan_tokens_emitted"] for r in results) / max(1, len(results)),
        "by_level": {k: sum(v) / len(v) for k, v in by_level.items()},
        "by_subject": {k: sum(v) / len(v) for k, v in by_subject.items()},
    }


def load_local_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(problem: str, prompt_style: str) -> str:
    if prompt_style == "train_compatible":
        return f"{problem}\n\n"
    if prompt_style == "instruction":
        return INSTRUCTION_PROMPT_TEMPLATE.format(problem=problem)
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def infer_plan_token_delta_path(
    lora_checkpoint: str | None,
    proj_checkpoint: str,
    explicit_path: str | None,
) -> str | None:
    if explicit_path:
        return explicit_path
    candidates: list[Path] = []
    if lora_checkpoint:
        candidates.append(Path(lora_checkpoint).resolve().parent / "plan_token_delta.pt")
    candidates.append(Path(proj_checkpoint).resolve().parent / "plan_token_delta.pt")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def next_token_from_outputs(outputs, plan_token_id: int, plan_token_delta: torch.Tensor | None) -> torch.Tensor:
    logits = apply_plan_token_logit_delta(
        logits=outputs.logits,
        hidden_states=outputs.hidden_states[-1],
        plan_token_id=plan_token_id,
        plan_token_delta=plan_token_delta,
    )
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


def load_math_test_split(dataset_name: str, split: str, dataset_config: str | None):
    tried: list[tuple[str, str | None, str]] = []
    candidates = [dataset_name]
    for fallback in ["EleutherAI/hendrycks_math", "hendrycks/competition_math"]:
        if fallback not in candidates:
            candidates.append(fallback)

    for name in candidates:
        config = dataset_config if name == dataset_name else None
        try:
            if config:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            print(f"Loaded dataset: name={name}, split={split}, config={config}")
            return ds
        except Exception as exc:  # noqa: BLE001
            tried.append((name, config, str(exc)))

    details = "\n".join(
        f"- name={name}, split={split}, config={config}: {err}" for name, config, err in tried
    )
    raise RuntimeError(
        "Failed to load MATH dataset from provided name and fallbacks.\n"
        f"Tried:\n{details}"
    )


def _next_token_forward(
    model,
    token_id: torch.Tensor,
    past_key_values,
    inputs_embeds: torch.Tensor | None = None,
    output_hidden_states: bool = False,
):
    if inputs_embeds is not None:
        return model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    return model(
        input_ids=token_id,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )


def autonomous_generate(
    model,
    tokenizer,
    proj,
    prompt: str,
    max_new_tokens: int,
    plan_token_id: int,
    plan_token_delta: torch.Tensor | None = None,
) -> tuple[str, int, int]:
    device = next(model.parameters()).device
    embed_dtype = model.get_input_embeddings().weight.dtype
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, use_cache=True, output_hidden_states=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated: list[int] = []
        plan_count = 0

        for _ in range(max_new_tokens):
            token_val = int(next_token.item())
            if token_val == plan_token_id:
                past_before_plan = past_key_values
                plan_embed = model.get_input_embeddings()(next_token).to(dtype=embed_dtype)
                if plan_token_delta is not None:
                    plan_embed = plan_embed + plan_token_delta.to(device=device, dtype=embed_dtype).unsqueeze(0).unsqueeze(1)

                plan_pass = _next_token_forward(
                    model,
                    next_token,
                    past_before_plan,
                    inputs_embeds=plan_embed,
                    output_hidden_states=True,
                )
                h_plan = plan_pass.hidden_states[-1][:, -1, :]
                t_i = proj(h_plan.to(dtype=next(proj.parameters()).dtype)).to(dtype=embed_dtype)
                planned_pass = _next_token_forward(
                    model,
                    next_token,
                    past_before_plan,
                    inputs_embeds=plan_embed + t_i.unsqueeze(1),
                    output_hidden_states=True,
                )
                outputs = planned_pass
                past_key_values = outputs.past_key_values
                plan_count += 1
            else:
                outputs = _next_token_forward(
                    model,
                    next_token,
                    past_key_values,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values

            generated.append(token_val)
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
            if token_val == tokenizer.eos_token_id or has_complete_boxed(decoded):
                break
            next_token = next_token_from_outputs(outputs, plan_token_id, plan_token_delta)
    return tokenizer.decode(generated, skip_special_tokens=False), len(generated), plan_count


def external_controller_generate(
    model,
    tokenizer,
    proj,
    prompt: str,
    max_steps: int,
    max_step_tokens: int,
    plan_token_id: int,
    plan_token_delta: torch.Tensor | None = None,
) -> tuple[str, int, int]:
    device = next(model.parameters()).device
    embed_dtype = model.get_input_embeddings().weight.dtype
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    all_generated = ids
    plan_count = 0

    with torch.no_grad():
        for _ in range(max_steps):
            # Pass up to current context for cache.
            pref = model(input_ids=all_generated, use_cache=True, return_dict=True)
            past = pref.past_key_values

            # Inject plan on the [PLAN] token embedding and obtain updated cache.
            plan_embed = model.get_input_embeddings()(
                torch.tensor([[plan_token_id]], device=device, dtype=torch.long)
            ).to(dtype=embed_dtype)
            if plan_token_delta is not None:
                plan_embed = plan_embed + plan_token_delta.to(device=device, dtype=embed_dtype).unsqueeze(0).unsqueeze(1)
            h_plan = model(
                inputs_embeds=plan_embed,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            ).hidden_states[-1][:, -1, :]
            t_i = proj(h_plan.to(dtype=next(proj.parameters()).dtype)).to(dtype=embed_dtype)

            plan_out = model(
                inputs_embeds=plan_embed + t_i.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            step_past = plan_out.past_key_values
            next_token = plan_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            plan_count += 1

            step_tokens = []
            for _ in range(max_step_tokens):
                step_out = model(
                    input_ids=next_token,
                    past_key_values=step_past,
                    use_cache=True,
                    return_dict=True,
                )
                step_past = step_out.past_key_values
                tok = int(next_token.item())
                step_tokens.append(tok)
                text = tokenizer.decode(step_tokens, skip_special_tokens=False)
                if tok == tokenizer.eos_token_id or "\n\n" in text or has_complete_boxed(text):
                    break
                next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            append = torch.tensor(
                [[plan_token_id] + step_tokens], device=device, dtype=torch.long
            )
            all_generated = torch.cat([all_generated, append], dim=1)
            if has_complete_boxed(tokenizer.decode(all_generated[0], skip_special_tokens=False)):
                break
    text = tokenizer.decode(all_generated[0], skip_special_tokens=False)
    return text, int(all_generated.size(1) - ids.size(1)), plan_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset_name", default="EleutherAI/hendrycks_math")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--local_eval_path", default=None)
    parser.add_argument("--prompt_style", choices=["train_compatible", "instruction"], default="train_compatible")
    parser.add_argument("--lora_checkpoint", default=None)
    parser.add_argument("--proj_checkpoint", required=True)
    parser.add_argument("--plan_token_delta_checkpoint", default=None)
    parser.add_argument("--proj_type", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--inference_mode", choices=["autonomous", "external_controller"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_step_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    rank, world_size, local_rank = ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PLAN]"]})
    plan_token_id = tokenizer.convert_tokens_to_ids("[PLAN]")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    if args.lora_checkpoint:
        base = PeftModel.from_pretrained(base, args.lora_checkpoint)
        base = base.merge_and_unload()
    model = base.to(device).eval()

    proj = ProjectionModule(model.config.hidden_size, proj_type=args.proj_type).to(device)
    proj.load_state_dict(torch.load(args.proj_checkpoint, map_location=device))
    proj.eval()
    plan_token_delta = None
    plan_token_delta_path = infer_plan_token_delta_path(
        lora_checkpoint=args.lora_checkpoint,
        proj_checkpoint=args.proj_checkpoint,
        explicit_path=args.plan_token_delta_checkpoint,
    )
    if plan_token_delta_path:
        plan_token_delta = torch.load(plan_token_delta_path, map_location=device).to(
            device=device, dtype=model.get_input_embeddings().weight.dtype
        )
        if rank == 0:
            print(f"Loaded plan_token_delta: {plan_token_delta_path}")

    if args.local_eval_path:
        problems = load_local_jsonl(args.local_eval_path)
        if rank == 0:
            print(f"Loaded local eval file: {args.local_eval_path} ({len(problems)} samples)")
    else:
        ds = load_math_test_split(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            dataset_config=args.dataset_config,
        )
        problems = [dict(x) for x in ds]
    if args.max_samples is not None:
        problems = problems[: args.max_samples]
    local = shard(problems, rank, world_size)

    local_results = []
    for item in local:
        prompt = build_prompt(item["problem"], args.prompt_style)
        if args.inference_mode == "autonomous":
            generated, generated_tokens, plan_emitted = autonomous_generate(
                model=model,
                tokenizer=tokenizer,
                proj=proj,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                plan_token_id=plan_token_id,
                plan_token_delta=plan_token_delta,
            )
        else:
            generated, generated_tokens, plan_emitted = external_controller_generate(
                model=model,
                tokenizer=tokenizer,
                proj=proj,
                prompt=prompt,
                max_steps=args.max_steps,
                max_step_tokens=args.max_step_tokens,
                plan_token_id=plan_token_id,
                plan_token_delta=plan_token_delta,
            )

        pred = extract_final_answer(generated)
        if "solution" in item:
            gold = extract_final_answer(str(item["solution"])) or str(item["solution"]).strip()
        else:
            gold = str(item.get("ground_truth_answer", "")).strip()
        local_results.append(
            {
                "problem": item["problem"],
                "subject": item.get("type", item.get("subject", "unknown")),
                "level": str(item.get("level", "unknown")),
                "prediction": pred,
                "gold": gold,
                "correct": grade_answer(pred, gold),
                "generated_tokens": generated_tokens,
                "plan_tokens_emitted": plan_emitted,
                "generated_text": generated,
            }
        )

    gathered: list[list[dict]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]

    if rank == 0:
        merged = [x for part in gathered for x in part]
        payload = {"metrics": summarize(merged), "results": merged}
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(payload["metrics"], indent=2))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

