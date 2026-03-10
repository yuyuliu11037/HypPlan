from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.dataset import (
    ReasoningPath,
    build_prefix_text,
    load_gsm8k_aug_dataset,
)
from src.model.hyperbolic import IdentityProjection
from src.model.planning_head import PlanningHead
from src.training.eval_utils import EvalRow, score_prediction, write_eval_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base model or trained adapter.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="YAML config path.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA adapter path. For EM mode, can point to checkpoint dir or lora_adapter dir.",
    )
    parser.add_argument(
        "--planning_head_path",
        type=str,
        default=None,
        help="Optional planning_head.pt path for EM mode. If omitted, inferred from adapter path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "em"],
        default="sft",
        help="sft: standard generation; em: dynamic planning-vector generation.",
    )
    parser.add_argument("--output", type=str, default="outputs/eval/baseline_predictions.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override evaluation split from config (for example: validation or test).",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_eval_data(cfg: Dict[str, Any], max_samples: int) -> List[ReasoningPath]:
    dataset_name = cfg["dataset_name"]
    eval_split = cfg.get("eval_split", "validation")
    data = load_gsm8k_aug_dataset(
        dataset_name=dataset_name,
        split=eval_split,
        max_samples=max_samples,
    )
    return data[:max_samples]


def build_prompt(question: str) -> str:
    return (
        "You are a math reasoning assistant.\n"
        "Solve the question step by step and end with the final answer.\n\n"
        f"Question:\n{question}\n\n"
        "Reasoning:\n"
    )


def generate_prediction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = out[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _resolve_em_paths(adapter_path: str, planning_head_path: str | None) -> Tuple[Path, Path]:
    adapter = Path(adapter_path).expanduser().resolve()
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter}")

    if planning_head_path:
        planning = Path(planning_head_path).expanduser().resolve()
        if not planning.exists():
            raise FileNotFoundError(f"planning_head_path does not exist: {planning}")
        return adapter, planning

    if adapter.name == "lora_adapter":
        planning = adapter.parent / "planning_head.pt"
        if not planning.exists():
            raise FileNotFoundError(
                f"EM mode requires planning_head.pt next to adapter. Missing: {planning}"
            )
        return adapter, planning

    lora_subdir = adapter / "lora_adapter"
    planning = adapter / "planning_head.pt"
    if lora_subdir.exists() and planning.exists():
        return lora_subdir, planning

    raise ValueError(
        "Unable to infer EM checkpoint contract. Provide --adapter_path as either "
        "`.../lora_adapter` or checkpoint dir containing `lora_adapter/` and `planning_head.pt`, "
        "or pass --planning_head_path explicitly."
    )


@torch.no_grad()
def _compute_plan_latent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    planning_head: PlanningHead,
    projection: IdentityProjection,
    question: str,
    prior_steps: List[str],
    max_question_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    prefix_text = build_prefix_text(question=question, steps=prior_steps, step_index=len(prior_steps))
    tokenized = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_question_tokens,
        return_tensors="pt",
        add_special_tokens=True,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    outputs = model(**tokenized, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    return projection(planning_head(hidden))


def _compose_inputs_with_plans(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    generated_ids: List[int],
    injection_points: List[int],
    plan_embeds: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    model_for_embeds = getattr(model, "module", model)
    prompt_embeds = model_for_embeds.get_input_embeddings()(prompt_ids)
    if generated_ids:
        gen_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        gen_embeds = model_for_embeds.get_input_embeddings()(gen_tensor)
    else:
        hidden = prompt_embeds.size(-1)
        gen_embeds = torch.empty((1, 0, hidden), dtype=prompt_embeds.dtype, device=device)

    parts: List[torch.Tensor] = [prompt_embeds]
    plan_idx = 0
    gen_len = len(generated_ids)
    for token_pos in range(gen_len + 1):
        while plan_idx < len(injection_points) and injection_points[plan_idx] == token_pos:
            parts.append(plan_embeds[plan_idx].view(1, 1, -1).to(dtype=prompt_embeds.dtype, device=device))
            plan_idx += 1
        if token_pos < gen_len:
            parts.append(gen_embeds[:, token_pos : token_pos + 1, :])
    return torch.cat(parts, dim=1)


@torch.no_grad()
def generate_with_em_planning(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    planning_head: PlanningHead,
    projection: IdentityProjection,
    question: str,
    max_question_tokens: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    prompt = build_prompt(question)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    generated_ids: List[int] = []
    injection_points: List[int] = [0]  # t0 inserted before first generated token
    prior_steps: List[str] = []
    current_step_text = ""
    t0 = _compute_plan_latent(
        model=model,
        tokenizer=tokenizer,
        planning_head=planning_head,
        projection=projection,
        question=question,
        prior_steps=prior_steps,
        max_question_tokens=max_question_tokens,
        device=device,
    )
    plan_embeds: List[torch.Tensor] = [t0]

    do_sample = temperature > 0.0
    for _ in range(max_new_tokens):
        combined_embeds = _compose_inputs_with_plans(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            injection_points=injection_points,
            plan_embeds=plan_embeds,
            device=device,
        )
        attention_mask = torch.ones((1, combined_embeds.size(1)), dtype=torch.long, device=device)
        outputs = model(inputs_embeds=combined_embeds, attention_mask=attention_mask, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(next_logits / max(temperature, 1e-5), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        generated_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

        piece = tokenizer.decode([next_id], skip_special_tokens=True)
        current_step_text += piece
        if "." in piece:
            # Regenerate planning vector after each finished step.
            completed = current_step_text.strip()
            if completed:
                prior_steps.append(completed)
            current_step_text = ""
            next_plan = _compute_plan_latent(
                model=model,
                tokenizer=tokenizer,
                planning_head=planning_head,
                projection=projection,
                question=question,
                prior_steps=prior_steps,
                max_question_tokens=max_question_tokens,
                device=device,
            )
            injection_points.append(len(generated_ids))
            plan_embeds.append(next_plan)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.split:
        cfg["eval_split"] = args.split
    data = load_eval_data(cfg, max_samples=args.max_samples)

    model_name_or_path = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    planning_head = None
    projection = None
    if args.mode == "em":
        if not args.adapter_path:
            raise ValueError("EM mode requires --adapter_path.")
        adapter_dir, planning_head_file = _resolve_em_paths(args.adapter_path, args.planning_head_path)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        hidden_size = model.config.hidden_size
        planning_head = PlanningHead(
            hidden_size=hidden_size,
            planning_dim=hidden_size,
            mlp_hidden_dim=cfg["planning_mlp_hidden_dim"],
        )
        state_dict = torch.load(planning_head_file, map_location="cpu")
        planning_head.load_state_dict(state_dict)
        projection = IdentityProjection()
    elif args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model = model.to(device)
    model.eval()
    if planning_head is not None:
        planning_head = planning_head.to(device)
        # Match model dtype (e.g. bf16) to avoid mat1/mat2 dtype mismatch in planning_head.
        planning_head = planning_head.to(dtype=next(model.parameters()).dtype)
        planning_head.eval()
    if projection is not None:
        projection = projection.to(device)
        projection.eval()

    rows: List[EvalRow] = []
    for sample in data:
        reference = sample.answer
        if args.mode == "em":
            assert planning_head is not None
            assert projection is not None
            prediction = generate_with_em_planning(
                model=model,
                tokenizer=tokenizer,
                planning_head=planning_head,
                projection=projection,
                question=sample.question,
                max_question_tokens=cfg["max_question_tokens"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )
        else:
            prompt = build_prompt(sample.question)
            prediction = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )
        scores = score_prediction(reference=reference, prediction=prediction)
        rows.append(
            EvalRow(
                sample_id=sample.sample_id,
                question=sample.question,
                reference=reference,
                prediction=prediction,
                exact_match=scores["exact_match"],
                reference_substring_match=scores["reference_substring_match"],
                last_number_match=scores["last_number_match"],
            )
        )

    metrics = write_eval_results(rows, Path(args.output))
    print("Evaluation completed.")
    print(f"Mode: {args.mode}")
    print(f"Predictions written to: {args.output}")
    print("Metrics:")
    for key, value in metrics.items():
        if key == "count":
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
