from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from src.evaluation.math_grading import extract_boxed, grade_answer
from src.evaluation.math_dataset import load_math_test
from src.model.planning_model import PlanningQwen


def setup_distributed() -> tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return rank, world_size
    return 0, 1


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_prompt(problem: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {problem}\n"
        "Solution:\n"
    )


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(int(x["correct"]) for x in results)

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in results:
        level = row.get("level", "unknown")
        subject = row.get("subject", "unknown")
        by_level[level]["total"] += 1
        by_level[level]["correct"] += int(row["correct"])
        by_subject[subject]["total"] += 1
        by_subject[subject]["correct"] += int(row["correct"])

    def finalize(bucket):
        out = {}
        for k, v in bucket.items():
            out[k] = {
                "accuracy": v["correct"] / max(v["total"], 1),
                "correct": v["correct"],
                "total": v["total"],
            }
        return out

    return {
        "overall_accuracy": correct / max(total, 1),
        "overall_correct": correct,
        "overall_total": total,
        "avg_steps_generated": sum(x["steps_generated"] for x in results) / max(total, 1),
        "avg_segments_detected": sum(x["segments_detected"] for x in results) / max(total, 1),
        "by_level": finalize(by_level),
        "by_subject": finalize(by_subject),
    }


def _detect_newline_boundary(decoded_text: str) -> bool:
    return decoded_text.endswith("\n\n")


def generate_one_step(
    model: PlanningQwen,
    tokenizer,
    context_ids: torch.Tensor,
    plan_token_id: int,
    max_step_tokens: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    plan_input = torch.cat(
        [context_ids, torch.tensor([[plan_token_id]], device=device, dtype=torch.long)],
        dim=1,
    )

    with torch.no_grad():
        out = model.base_model(
            input_ids=plan_input,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        h_plan = hidden[:, -1, :]
        t_i = model.proj(h_plan)
        hidden[:, -1, :] = hidden[:, -1, :] + t_i
        logits = model.base_model.lm_head(hidden)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    generated = [int(next_token.item())]
    running = torch.cat([plan_input, next_token], dim=1)

    for _ in range(max_step_tokens - 1):
        with torch.no_grad():
            out = model.base_model(input_ids=running, use_cache=False, return_dict=True)
            step_next = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(int(step_next.item()))
        running = torch.cat([running, step_next], dim=1)

        text = tokenizer.decode(generated, skip_special_tokens=False)
        if _detect_newline_boundary(text):
            break

    return generated, t_i.squeeze(0)


def estimate_segments(model: PlanningQwen, plan_vectors: list[torch.Tensor]) -> int:
    if not plan_vectors:
        return 0
    vec = torch.stack(plan_vectors, dim=0)

    if model.segment_classifier is not None:
        logits = model.segment_classifier(vec)
        pred = torch.argmax(logits, dim=-1)
        return int((pred[1:] != pred[:-1]).sum().item()) + 1

    # Contrastive mode fallback: count depth-score resets as segment boundaries.
    half = vec.size(-1) // 2
    scores = model.depth_readout(vec[:, half:]).squeeze(-1)
    return int((scores[1:] < scores[:-1]).sum().item()) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--proj_checkpoint", type=str, required=True)
    parser.add_argument("--proj_type", type=str, default="mlp", choices=["linear", "mlp"])
    parser.add_argument(
        "--structural_loss",
        type=str,
        default="simple",
        choices=["simple", "contrastive"],
    )
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_step_tokens", type=int, default=256)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PLAN]"]})
    plan_token_id = tokenizer.convert_tokens_to_ids("[PLAN]")

    model = PlanningQwen(
        model_name=args.model_name,
        proj_type=args.proj_type,
        structural_loss=args.structural_loss,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.load_trainable_state_dict(torch.load(args.proj_checkpoint, map_location="cpu"))
    model.eval()

    ds = load_math_test()
    all_items = [dict(x) for x in ds]
    shard = [x for i, x in enumerate(all_items) if i % world_size == rank]

    local_results = []
    iterator = tqdm(
        shard,
        disable=(rank != 0),
        mininterval=60.0,
        miniters=50,
    )
    for item in iterator:
        prompt = build_prompt(item["problem"])
        context_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        step_count = 0
        plan_vectors: list[torch.Tensor] = []
        for _ in range(args.max_steps):
            step_tokens, t_i = generate_one_step(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                plan_token_id=plan_token_id,
                max_step_tokens=args.max_step_tokens,
                device=device,
            )
            step_count += 1
            plan_vectors.append(t_i.detach().float().cpu())

            appended = torch.tensor(step_tokens, dtype=torch.long, device=device).unsqueeze(0)
            context_ids = torch.cat(
                [
                    context_ids,
                    torch.tensor([[plan_token_id]], dtype=torch.long, device=device),
                    appended,
                ],
                dim=1,
            )

            decoded = tokenizer.decode(context_ids[0], skip_special_tokens=True)
            if extract_boxed(decoded) is not None:
                break

        decoded = tokenizer.decode(context_ids[0], skip_special_tokens=True)
        pred = extract_boxed(decoded)
        gt = extract_boxed(item.get("solution", ""))
        correct = grade_answer(pred, gt)
        segments_detected = estimate_segments(model, plan_vectors)

        local_results.append(
            {
                "problem": item["problem"],
                "prediction": pred,
                "ground_truth": gt,
                "correct": correct,
                "level": item.get("level", "unknown"),
                "subject": item.get("type", "unknown"),
                "steps_generated": step_count,
                "segments_detected": segments_detected,
            }
        )

    gathered = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
        results = [x for part in gathered for x in part]
    else:
        results = local_results

    if rank == 0:
        metrics = compute_metrics(results)
        payload = {"metrics": metrics, "results": results}
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(json.dumps(metrics, indent=2))

    cleanup_distributed()


if __name__ == "__main__":
    main()
