"""Tree-of-Thoughts (Yao et al. 2023) baseline for Game of 24.

Paper-faithful BFS: at each of 3 arithmetic steps, for every current trajectory
we call a GENERATOR model with the propose_prompt to produce candidate next
ops, then call an EVALUATOR model with the value_prompt 3× per candidate to
score it (sum of weighted sure/likely/impossible labels), then keep the top-5
by score. After 3 steps, a trajectory is "correct" if it ends in "(left: 24)"
and its ops validate as a legal 3-step Game-of-24 solution.

Supports both any-of-top-K (matches the paper's 74% metric) and top-1 greedy-
equivalent reporting. Multi-seed support via --seed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate_24 import parse_and_validate


# --- Paper-verbatim prompts from github.com/princeton-nlp/tree-of-thought-llm ---

PROPOSE_PROMPT = """Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""

VALUE_PROMPT = """Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
"""


# --- Parsing ---

# Chat models (Qwen-Instruct etc.) often emit prose with backticks around code
# and use "leaving" instead of "left". Normalize before regex.
def _normalize_cand_text(text: str) -> str:
    t = text
    # Drop backticks and asterisks (markdown emphasis)
    t = t.replace("`", "").replace("**", "").replace("*", "*")
    # "leaving" → "left" so we only need one keyword
    t = re.sub(r"\(leaving:", "(left:", t, flags=re.IGNORECASE)
    t = re.sub(r"\(remaining:", "(left:", t, flags=re.IGNORECASE)
    t = re.sub(r"\(left over:", "(left:", t, flags=re.IGNORECASE)
    return t


# Candidate line: "a op b = c (left: x y z)". Accepts ints, fractions, decimals,
# and optional negatives. Multi-line findall.
CAND_RE = re.compile(
    r"(-?[\d./]+)\s*([+\-*/])\s*(-?[\d./]+)\s*=\s*(-?[\d./]+)\s*\(left:\s*([^)]*)\)"
)

# SFT-leak format: "Step N: a op b = c. Remaining: x y z"
CAND_RE_SFT = re.compile(
    r"Step\s+\d+:\s*(-?[\d./]+)\s*([+\-*/])\s*(-?[\d./]+)\s*=\s*(-?[\d./]+)\.\s*Remaining:\s*([0-9 .]+)"
)


def parse_candidates(text: str) -> list[dict]:
    """Extract (line, remaining_str) pairs from a generator output.

    Accepts both ToT format and SFT-leak format. Deduplicates by line text.
    Chat-model verbosity (backticks, 'leaving', markdown) is normalized first.
    """
    text = _normalize_cand_text(text)
    out: list[dict] = []
    seen: set = set()
    for m in CAND_RE.finditer(text):
        a, op, b, r, left = m.groups()
        left = left.strip()
        line = f"{a} {op} {b} = {r} (left: {left})"
        if line in seen:
            continue
        seen.add(line)
        out.append({"line": line, "remaining": left})
    for m in CAND_RE_SFT.finditer(text):
        a, op, b, r, left = m.groups()
        left = left.strip()
        line = f"{a} {op} {b} = {r} (left: {left})"
        if line in seen:
            continue
        seen.add(line)
        out.append({"line": line, "remaining": left})
    return out


LEFT_RE = re.compile(r"\(left:\s*([^)]*)\)")


def get_current_numbers(trajectory: str, initial: str) -> str:
    """Extract last '(left: X Y Z)' contents from trajectory. Fallback to initial."""
    matches = LEFT_RE.findall(trajectory)
    if matches:
        return matches[-1].strip()
    return initial


def value_score(text: str) -> float:
    """Paper value aggregation: sure=20, likely=1, impossible=0.001.

    Looks at the last non-empty line of `text` (one value-sample output).
    """
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return 0.0
    last = lines[-1].lower()
    if "impossible" in last:
        return 0.001
    if "likely" in last:
        return 1.0
    if "sure" in last:
        return 20.0
    return 0.0


# --- Trajectory → problem string (for final validation) ---

def trajectory_to_generation(problem: str, y: str) -> str:
    """Convert a ToT trajectory like "4 + 5 = 9 (left: 6 9 10)\\n..." to the
    "Step N:" format parse_and_validate() expects.
    """
    out_lines = []
    for idx, m in enumerate(CAND_RE.finditer(y), start=1):
        a, op, b, r, left = m.groups()
        left = left.strip()
        is_final_24 = False
        if idx == 3 and left.replace(".", "").replace("-", "").isdigit():
            try:
                is_final_24 = int(float(left)) == 24
            except (ValueError, OverflowError):
                is_final_24 = False
        if is_final_24:
            out_lines.append(f"Step {idx}: {a} {op} {b} = {r}. Answer: 24")
        else:
            out_lines.append(f"Step {idx}: {a} {op} {b} = {r}. Remaining: {left}")
    return "\n".join(out_lines)


# --- Chat-template wrapper for chat-tuned models (e.g. Qwen-Instruct) ---

_SYSTEM_TERSE = (
    "You are terse and precise. Follow the exact output format shown in the "
    "examples. Do NOT add backticks, markdown, prose, commentary, or bullet "
    "dashes. Output only the requested lines in the exact format."
)


def maybe_chat_wrap(tokenizer, raw_prompt: str, use_chat: bool) -> str:
    """If use_chat, wrap the raw ToT prompt as a user turn so a chat-tuned
    model will respond in assistant format. The raw prompt's few-shot
    exemplars stay intact — we just rebadge them as one user turn and let
    the assistant continue. A terse system message discourages markdown/
    backtick decorations that would trip the parser."""
    if not use_chat:
        return raw_prompt
    msgs = [
        {"role": "system", "content": _SYSTEM_TERSE},
        {"role": "user", "content": raw_prompt},
    ]
    # MistralCommonTokenizer (Mistral-Small 3.2+) doesn't accept
    # add_generation_prompt / enable_thinking kwargs; it already primes the
    # assistant turn in its raw template.
    try:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except (TypeError, ValueError):
        try:
            out = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except (TypeError, ValueError):
            out = tokenizer.apply_chat_template(msgs, tokenize=False)
    # Optional post-template suffix (e.g. "<|channel|>final<|message|>" to
    # force GPT-OSS into its final channel and skip the analysis thinking
    # block). Controlled via env var so no signature change is needed.
    suffix = os.environ.get("TOT_PROMPT_SUFFIX", "")
    if suffix:
        out = out + suffix
    return out


# --- Model plumbing ---

def load_model(name_or_path: str, device: torch.device,
               load_in_4bit: bool = False):
    print(f"  loading {name_or_path} (4bit={load_in_4bit})", flush=True)
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass  # MistralCommonTokenizer doesn't expose padding_side
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs = dict(quantization_config=bnb, device_map={"": device},
                      trust_remote_code=True)
    else:
        kwargs = dict(torch_dtype=torch.bfloat16, device_map={"": device},
                      trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs).eval()
    except ValueError as e:
        if "Unrecognized configuration" in str(e) and "Mistral3" in str(e):
            from transformers.models.mistral3 import Mistral3ForConditionalGeneration
            model = Mistral3ForConditionalGeneration.from_pretrained(
                name_or_path, **kwargs).eval()
        else:
            raise
    return tok, model


@torch.no_grad()
def batched_generate(model, tok, prompts: list[str], max_new_tokens: int,
                     temperature: float, n: int, device: torch.device,
                     batch_size: int = 16) -> list[list[str]]:
    """Return [len(prompts)] x [n] list of generated strings."""
    per_prompt: list[list[str]] = [[] for _ in prompts]
    for sample_i in range(n):
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start: start + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tok.pad_token_id,
            )
            gen = out[:, enc["input_ids"].size(1):]
            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            for i, text in enumerate(decoded):
                per_prompt[start + i].append(text)
    return per_prompt


# --- BFS loop ---

def tot_solve(problem: str, gen_tok, gen_model, eval_tok, eval_model,
              device: torch.device, n_generate: int, n_evaluate: int,
              n_select: int, temperature: float,
              propose_max_new: int = 200, value_max_new: int = 150,
              batch_size: int = 16, use_chat_template: bool = False) -> dict:
    initial = problem.replace(",", " ")
    ys: list[str] = [""]

    stats = {"per_step_candidates": [], "per_step_selected": []}

    for step in range(3):
        # Propose: one call per current y
        current_nums = [get_current_numbers(y, initial) for y in ys]
        propose_prompts = [
            maybe_chat_wrap(gen_tok, PROPOSE_PROMPT.format(input=nums),
                            use_chat_template)
            for nums in current_nums
        ]
        gen_outs = batched_generate(gen_model, gen_tok, propose_prompts,
                                    max_new_tokens=propose_max_new,
                                    temperature=temperature,
                                    n=n_generate, device=device,
                                    batch_size=batch_size)

        new_ys: list[str] = []
        for y, gen_samples in zip(ys, gen_outs):
            for text in gen_samples:
                for cand in parse_candidates(text):
                    new_y = y + cand["line"] + "\n"
                    new_ys.append(new_y)
        # Dedup
        seen_y: set = set()
        deduped: list[str] = []
        for y in new_ys:
            if y in seen_y:
                continue
            seen_y.add(y)
            deduped.append(y)
        new_ys = deduped
        stats["per_step_candidates"].append(len(new_ys))
        if not new_ys:
            break

        # Value: score each candidate with n_evaluate samples
        cand_remaining = [get_current_numbers(y, initial) for y in new_ys]
        value_prompts = [
            maybe_chat_wrap(eval_tok, VALUE_PROMPT.format(input=rem),
                            use_chat_template)
            for rem in cand_remaining
        ]
        v_outs = batched_generate(eval_model, eval_tok, value_prompts,
                                  max_new_tokens=value_max_new,
                                  temperature=temperature,
                                  n=n_evaluate, device=device,
                                  batch_size=batch_size)
        scores = [sum(value_score(s) for s in samples) for samples in v_outs]

        # Top-k
        ranked = sorted(zip(new_ys, scores), key=lambda p: -p[1])
        ys = [y for y, _ in ranked[:n_select]]
        stats["per_step_selected"].append(len(ys))

    return {"ys": ys, "stats": stats}


# --- Runner ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True)
    ap.add_argument("--evaluator", default=None,
                    help="Evaluator model path. Ignored if --shared_model.")
    ap.add_argument("--shared_model", action="store_true",
                    help="Use generator as both propose + value role "
                         "(evaluator arg ignored). Saves GPU memory.")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Wrap propose/value prompts in the model's chat "
                         "template (needed for chat-tuned models like "
                         "Qwen2.5-Instruct).")
    ap.add_argument("--test_data", default="data/24_test_tot.jsonl")
    ap.add_argument("--output_dir", default="results/tot_baseline/seed_1234")
    ap.add_argument("--n_generate", type=int, default=1)
    ap.add_argument("--n_evaluate", type=int, default=3)
    ap.add_argument("--n_select", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load base model(s) in 4-bit NF4 (for 32B on 48GB GPUs).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = Path(args.output_dir) / "generations.jsonl"
    out_metrics = Path(args.output_dir) / "metrics.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.shared_model:
        print(f"Loading shared model (used for both propose + value): "
              f"{args.generator}", flush=True)
        gen_tok, gen_model = load_model(args.generator, device,
                                        load_in_4bit=args.load_in_4bit)
        eval_tok, eval_model = gen_tok, gen_model
    else:
        print("Loading models (generator, evaluator)", flush=True)
        gen_tok, gen_model = load_model(args.generator, device,
                                        load_in_4bit=args.load_in_4bit)
        eval_tok, eval_model = load_model(args.evaluator, device,
                                          load_in_4bit=args.load_in_4bit)

    # Deduplicate test problems (file has one line per ground-truth trajectory)
    seen: set = set()
    problems: list[str] = []
    with open(args.test_data) as f:
        for line in f:
            p = json.loads(line)["problem"]
            if p not in seen:
                seen.add(p)
                problems.append(p)
    if args.limit > 0:
        problems = problems[: args.limit]
    print(f"Running ToT on {len(problems)} problems, seed={args.seed}",
          flush=True)

    n_any = 0
    n_top1 = 0
    total = 0
    t0 = time.time()
    with out_jsonl.open("w") as fout:
        for i, problem in enumerate(problems):
            total += 1
            res = tot_solve(problem, gen_tok, gen_model, eval_tok, eval_model,
                            device=device,
                            n_generate=args.n_generate,
                            n_evaluate=args.n_evaluate,
                            n_select=args.n_select,
                            temperature=args.temperature,
                            batch_size=args.batch_size,
                            use_chat_template=args.use_chat_template)
            ys = res["ys"]
            validities = [
                parse_and_validate(problem, trajectory_to_generation(problem, y))
                for y in ys
            ]
            any_correct = any(validities)
            top1_correct = bool(validities[0]) if validities else False
            n_any += int(any_correct)
            n_top1 += int(top1_correct)

            fout.write(json.dumps({
                "problem": problem,
                "ys": ys,
                "validities": validities,
                "any_correct": any_correct,
                "top1_correct": top1_correct,
                "stats": res["stats"],
            }) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0 or i == len(problems) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(problems) - (i + 1)) / rate
                print(f"  {i+1}/{len(problems)}  any={n_any/max(total,1):.3f}"
                      f"  top1={n_top1/max(total,1):.3f}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                      flush=True)

    metrics = {
        "seed": args.seed,
        "n_problems": total,
        "any_of_top_k_accuracy": n_any / max(total, 1),
        "top1_accuracy": n_top1 / max(total, 1),
        "n_any": n_any,
        "n_top1": n_top1,
        "n_select": args.n_select,
        "temperature": args.temperature,
        "generator": args.generator,
        "evaluator": args.evaluator,
    }
    with out_metrics.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n=== Done. any-of-{args.n_select}: "
          f"{metrics['any_of_top_k_accuracy']:.4f}  "
          f"top1: {metrics['top1_accuracy']:.4f} ===", flush=True)


if __name__ == "__main__":
    main()
