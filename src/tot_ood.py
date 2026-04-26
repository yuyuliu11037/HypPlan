"""Tree-of-Thoughts BFS for the 3 OOD tasks (ProntoQA / Blocksworld /
GraphColor).

Faithful to Yao et al. 2023 ToT BFS: at each depth, expand every current
state with `n_generate` propose calls (sampled at temperature>0 for
diversity), score each candidate with `n_evaluate` value calls, then keep
the top `n_select` by value-sum. After depth-budget steps, score the top
trajectory(ies) via task-specific scorers.

Same base model (Qwen2.5-14B-Instruct) used for both propose + value.

Adapter pattern: each task implements propose_prompt, value_prompt,
extract_step, is_terminal, is_correct.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Adapters ----------


class Adapter:
    name: str
    max_depth: int = 12

    def __init__(self, rec: dict) -> None:
        self.rec = rec

    def init_partial(self) -> str:
        return ""

    def propose_prompt(self, partial: str) -> str:
        raise NotImplementedError

    def value_prompt(self, partial: str) -> str:
        raise NotImplementedError

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        """Extract zero or more candidate next-step strings from a generation."""
        raise NotImplementedError

    def is_terminal(self, partial: str) -> bool:
        raise NotImplementedError

    def is_correct(self, partial: str) -> bool:
        raise NotImplementedError


# ---------- ProntoQA adapter ----------

_PQ_PROPOSE_HEADER = (
    "You will derive new facts step by step from the rules, then conclude "
    "with the answer (A=true, B=false). Each step uses one rule and one "
    "previously known fact.\n"
    "Propose 3 different sensible next-line candidates for the derivation "
    "below. Output exactly 3 lines (no numbering, no commentary). Each line "
    "must be either\n"
    "  Step N: <derived fact>.\n"
    "or\n"
    "  Answer: A\n"
    "or\n"
    "  Answer: B\n\n"
)

_PQ_VALUE_HEADER = (
    "Given the rules and the partial derivation, decide whether continuing "
    "this derivation will arrive at the correct answer. Reply with EXACTLY "
    "one of: sure / likely / impossible. No other text.\n\n"
)


def _pq_extract_rules_and_query(prompt: str) -> tuple[str, str, str]:
    """Pull (rules_text, init_fact_text, question_text) from PQ test record.

    Test prompt structure: 3 few-shot Context blocks ending with answers,
    then a final Context block ending with 'Answer:' (no answer).
    """
    last_ctx = prompt.rfind("Context: ")
    body = prompt[last_ctx + len("Context: "):]
    # body: "<rules>. <Subject is a Pred>. \nIs the following statement true or false? <Statement>.\nA) True B) False\nAnswer:"
    rules_section, _, after = body.partition(
        "\nIs the following statement true or false? ")
    statement = after.split("\n", 1)[0].rstrip(".? ")
    # split off the last "Subject is a Pred" sentence as init_fact
    sentences = re.split(r"(?<=[.])\s+", rules_section.strip())
    if len(sentences) >= 2:
        rules_text = " ".join(sentences[:-1])
        init_fact = sentences[-1].rstrip(".")
    else:
        rules_text = rules_section.strip()
        init_fact = ""
    return rules_text, init_fact, statement


class ProntoQAAdapter(Adapter):
    name = "pq"
    max_depth = 12

    def __init__(self, rec: dict) -> None:
        super().__init__(rec)
        self.rules, self.init_fact, self.question = (
            _pq_extract_rules_and_query(rec["prompt"]))

    def _common(self) -> str:
        return (f"Rules:\n{self.rules}\n\n"
                f"Initial fact: {self.init_fact}.\n"
                f"Question: is the following true? {self.question}\n\n")

    def propose_prompt(self, partial: str) -> str:
        body = self._common()
        n_so_far = partial.count("Step ")
        next_n = n_so_far + 1
        if partial:
            body += "Derivation so far:\n" + partial
        else:
            body += "Derivation so far: (none)\n"
        body += f"\nStep {next_n}:"
        return _PQ_PROPOSE_HEADER + body

    def value_prompt(self, partial: str) -> str:
        body = self._common()
        body += "Partial derivation:\n" + (partial if partial else "(none)\n")
        body += "\nEvaluation:"
        return _PQ_VALUE_HEADER + body

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        out: list[str] = []
        for ln in gen.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r"^[\-\*\d\.]+\s*", "", ln)
            if re.match(r"^Answer:\s*[AB]", ln, re.I):
                m2 = re.search(r"[AB]", ln)
                if m2:
                    out.append(f"Answer: {m2.group(0).upper()}")
                    continue
            if re.match(r"^Step\s+\d+:", ln):
                out.append(ln.rstrip())
        return out

    def is_terminal(self, partial: str) -> bool:
        return bool(re.search(r"Answer:\s*[AB]", partial))

    def is_correct(self, partial: str) -> bool:
        gold = self.rec["answer_label"]
        m = re.search(r"Answer:\s*([AB])", partial)
        if not m:
            return False
        return m.group(1) == gold


# ---------- Blocksworld adapter ----------

_BW_PROPOSE_HEADER = (
    "Propose 3 different sensible next Blocksworld actions for the partial "
    "plan below. Each action MUST be valid in the current state. Use the "
    "literal forms:\n"
    "  pick-up X\n  put-down X\n  stack X Y\n  unstack X Y\n"
    "where X,Y are colors (red, blue, orange, yellow). Output exactly 3 "
    "lines (one action per line, no numbering, no commentary).\n\n"
)

_BW_VALUE_HEADER = (
    "Given the Blocksworld initial state, the goal, and the partial action "
    "sequence, decide whether continuing this plan will reach the goal. "
    "Reply with EXACTLY one of: sure / likely / impossible. No other text.\n\n"
)


class BlocksworldAdapter(Adapter):
    name = "bw"
    max_depth = 14

    def __init__(self, rec: dict) -> None:
        super().__init__(rec)
        from src.oracle_blocksworld import parse_problem
        try:
            self.problem = parse_problem(rec["prompt"])
        except Exception:
            self.problem = None
        # Render a clean init/goal description (re-extracted from prompt) for
        # the propose+value prompts.
        last_stmt = rec["prompt"].split("[STATEMENT]")[-1]
        init_text, _, goal_text = last_stmt.partition(
            "My goal is to have that")
        goal_text = goal_text.split("My plan is")[0]
        self.init_text = init_text.replace(
            "As initial conditions I have that, ", "Initially, ").strip()
        self.goal_text = "Goal: " + goal_text.strip().rstrip(".")

    def _common(self) -> str:
        return f"{self.init_text}\n{self.goal_text}\n\n"

    def _action_lines(self, partial: str) -> list[str]:
        return [ln.strip() for ln in partial.splitlines() if ln.strip()]

    def _simulate(self, partial: str):
        from src.oracle_blocksworld import (apply_action, is_goal, Action)
        if self.problem is None:
            return None, False, False
        state = self.problem.init
        seen_states = {state}
        legal = True
        for ln in self._action_lines(partial):
            ln = ln.strip().strip("()")
            parts = ln.split()
            if not parts:
                continue
            op = parts[0]
            args = tuple(parts[1:])
            new_state = apply_action(state, Action(op=op, args=args))
            if new_state == state:
                legal = False
                return state, False, legal
            if new_state in seen_states:
                # Visited cycle — terminate this branch as a dead-end so the
                # search can pick a different continuation.
                legal = False
                return state, False, legal
            seen_states.add(new_state)
            state = new_state
            if is_goal(state, self.problem.goal):
                return state, True, legal
        return state, False, legal

    def _render_current_state(self, partial: str) -> str:
        from src.oracle_blocksworld import render_state
        if self.problem is None:
            return ""
        state, _, _ = self._simulate(partial)
        return render_state(self.problem, state)

    def propose_prompt(self, partial: str) -> str:
        body = self._common()
        body += self._render_current_state(partial) + "\n\n"
        if partial:
            body += "Plan so far:\n" + partial
        else:
            body += "Plan so far: (empty)\n"
        body += "\nNext action:"
        return _BW_PROPOSE_HEADER + body

    def value_prompt(self, partial: str) -> str:
        body = self._common()
        body += self._render_current_state(partial) + "\n\n"
        body += "Plan so far:\n" + (partial if partial else "(empty)\n")
        body += "\nEvaluation:"
        return _BW_VALUE_HEADER + body

    def extract_step(self, gen: str, partial: str) -> Optional[str]:
        # Take first action-shaped line: "op X" or "op X Y" or "(op X Y)"
        for ln in gen.splitlines():
            ln = ln.strip().strip("()")
            ln = re.sub(r"^[\s\-\*\d\.]+", "", ln)
            # Natural language fallback first
            if m := re.search(
                r"unstack the (\w+) block from(?: on top of| from)? the (\w+) block",
                ln, re.I):
                return f"unstack {m.group(1).lower()} {m.group(2).lower()}"
            if m := re.search(
                r"stack the (\w+) block on(?:to| top of)? the (\w+) block",
                ln, re.I):
                return f"stack {m.group(1).lower()} {m.group(2).lower()}"
            if m := re.search(r"pick(?:[ -])up the (\w+) block", ln, re.I):
                return f"pick-up {m.group(1).lower()}"
            if m := re.search(r"put(?: down|-down)? the (\w+) block", ln, re.I):
                return f"put-down {m.group(1).lower()}"
            # Direct
            m = re.match(
                r"^(pick-up|put-down|stack|unstack)\s+(\w+)(?:\s+(\w+))?\b",
                ln)
            if m:
                op = m.group(1)
                if m.group(3):
                    return f"{op} {m.group(2)} {m.group(3)}"
                return f"{op} {m.group(2)}"
        return None

    def is_terminal(self, partial: str) -> bool:
        _, reached, legal = self._simulate(partial)
        # Treat illegal-action trajectories as terminal so we stop expanding
        # them (their next propose would just dig deeper into nonsense).
        return reached or not legal

    def is_correct(self, partial: str) -> bool:
        _, reached, _ = self._simulate(partial)
        return reached


# ---------- GraphColor adapter ----------

_GC_PROPOSE_HEADER = (
    "Output exactly ONE assignment line of the form 'V<i> = <color>' for the "
    "next uncolored vertex, where color is one of red/green/blue. One line "
    "only. No commentary, no extra text.\n\n"
)

_GC_VALUE_HEADER = (
    "Given the 3-coloring problem and partial coloring, decide whether the "
    "partial coloring is valid (no edge conflict so far) AND can be extended "
    "to a complete valid coloring. Reply with EXACTLY one of: "
    "sure / likely / impossible. No other text.\n\n"
)


class GraphColorAdapter(Adapter):
    name = "gc"

    def __init__(self, rec: dict) -> None:
        super().__init__(rec)
        from src.oracle_graphcolor import Problem
        self.problem = Problem(n=rec["n"],
                                edges=tuple(map(tuple, rec["edges"])))
        self.max_depth = self.problem.n

    def _common(self) -> str:
        edges = ", ".join(f"(V{u},V{v})" for u, v in self.problem.edges)
        return (f"Graph 3-coloring task.\n"
                f"Vertices: {', '.join('V'+str(i) for i in range(self.problem.n))}\n"
                f"Edges: {edges}\n"
                f"Adjacent vertices must have different colors. "
                f"Colors are red, green, blue.\n\n")

    def _coloring_so_far(self, partial: str) -> dict[int, str]:
        from src.oracle_graphcolor import parse_coloring
        return parse_coloring(partial, self.problem)

    def _next_uncolored(self, partial: str) -> int:
        c = self._coloring_so_far(partial)
        for i in range(self.problem.n):
            if i not in c:
                return i
        return -1

    def propose_prompt(self, partial: str) -> str:
        body = self._common()
        nxt = self._next_uncolored(partial)
        if partial:
            body += "Coloring so far:\n" + partial
        else:
            body += "Coloring so far: (none)\n"
        body += f"\nV{nxt} ="
        return _GC_PROPOSE_HEADER + body

    def value_prompt(self, partial: str) -> str:
        body = self._common()
        body += "Coloring so far:\n" + (partial if partial else "(none)\n")
        body += "\nEvaluation:"
        return _GC_VALUE_HEADER + body

    def extract_step(self, gen: str, partial: str) -> Optional[str]:
        nxt = self._next_uncolored(partial)
        for ln in gen.splitlines():
            ln = ln.strip()
            m = re.search(r"V?(\d+)?\s*[:=]\s*(red|green|blue|R|G|B)\b",
                           ln, re.I)
            if m:
                v = int(m.group(1)) if m.group(1) else nxt
                if v != nxt:
                    continue
                color_full = {"R": "red", "G": "green", "B": "blue"}.get(
                    m.group(2).upper()[0], m.group(2).lower())
                return f"V{v} = {color_full}"
        return None

    def is_terminal(self, partial: str) -> bool:
        c = self._coloring_so_far(partial)
        return len(c) >= self.problem.n

    def is_correct(self, partial: str) -> bool:
        from src.oracle_graphcolor import score_coloring
        c = self._coloring_so_far(partial)
        return score_coloring(self.problem, c)


ADAPTERS = {"pq": ProntoQAAdapter, "bw": BlocksworldAdapter,
             "gc": GraphColorAdapter}


# ---------- Generic ToT BFS ----------


def value_score(text: str) -> float:
    """sure=20, likely=1, impossible=0.001 — same as ToT paper."""
    t = text.strip().lower().replace("\n", " ")
    # Take last 200 chars to avoid earlier context
    t = t[-200:]
    if "impossible" in t:
        return 0.001
    if "sure" in t:
        return 20.0
    if "likely" in t:
        return 1.0
    return 0.0


@torch.no_grad()
def batched_generate(model, tok, prompts: list[str], max_new_tokens: int,
                      temperature: float, n: int, device: torch.device,
                      batch_size: int = 8) -> list[list[str]]:
    per_prompt: list[list[str]] = [[] for _ in prompts]
    for _ in range(n):
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


def _chat_wrap(tok, raw: str) -> str:
    msgs = [
        {"role": "system",
          "content": ("You are precise. Follow the exact output format the "
                      "user requests. Do not add explanations or markdown.")},
        {"role": "user", "content": raw},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    except Exception:
        return raw


def tot_solve(adapter: Adapter, tok, model, device,
              n_generate: int, n_evaluate: int, n_select: int,
              temperature: float, propose_max_new: int,
              value_max_new: int, batch_size: int) -> dict:
    ys: list[str] = [adapter.init_partial()]
    history: list[dict] = []

    for depth in range(adapter.max_depth):
        # Drop terminal trajectories from the frontier (keep them aside).
        terminal: list[str] = [y for y in ys if adapter.is_terminal(y)]
        active: list[str] = [y for y in ys if not adapter.is_terminal(y)]
        if not active:
            ys = terminal
            break

        # Propose
        propose_prompts = [_chat_wrap(tok, adapter.propose_prompt(y))
                            for y in active]
        gen_outs = batched_generate(
            model, tok, propose_prompts, max_new_tokens=propose_max_new,
            temperature=temperature, n=n_generate, device=device,
            batch_size=batch_size)

        new_ys: list[str] = []
        for y, gen_samples in zip(active, gen_outs):
            for text in gen_samples:
                step = adapter.extract_step(text, y)
                if step is None:
                    continue
                if y and not y.endswith("\n"):
                    new_y = y + "\n" + step
                else:
                    new_y = (y or "") + step
                new_ys.append(new_y)
        # Dedup
        seen: set = set()
        deduped: list[str] = []
        for y in new_ys:
            if y in seen:
                continue
            seen.add(y)
            deduped.append(y)
        new_ys = deduped + terminal  # carry forward terminal trajectories
        if not new_ys:
            ys = terminal
            break

        # Value
        value_prompts = [_chat_wrap(tok, adapter.value_prompt(y))
                          for y in new_ys]
        v_outs = batched_generate(
            model, tok, value_prompts, max_new_tokens=value_max_new,
            temperature=temperature, n=n_evaluate, device=device,
            batch_size=batch_size)
        scores = [sum(value_score(s) for s in samples) for samples in v_outs]

        # Top-N select
        ranked = sorted(zip(new_ys, scores), key=lambda p: -p[1])
        ys = [y for y, _ in ranked[:n_select]]
        history.append({"depth": depth, "n_candidates": len(new_ys),
                          "n_selected": len(ys),
                          "top_scores": [s for _, s in ranked[:n_select]]})

    return {"ys": ys, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n_generate", type=int, default=3)
    ap.add_argument("--n_evaluate", type=int, default=3)
    ap.add_argument("--n_select", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--propose_max_new", type=int, default=80)
    ap.add_argument("--value_max_new", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={"": device}).eval()

    records = [json.loads(l) for l in open(args.test_data)]
    if args.limit > 0:
        records = records[: args.limit]
    if args.shard_world > 1:
        records = records[args.shard_rank :: args.shard_world]
    print(f"ToT task={args.task} on {len(records)} records "
           f"(shard {args.shard_rank}/{args.shard_world})", flush=True)

    AdapterCls = ADAPTERS[args.task]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_top1 = 0
    n_any = 0
    total = 0
    t0 = time.time()
    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            adapter = AdapterCls(rec)
            try:
                res = tot_solve(
                    adapter, tok, model, device,
                    n_generate=args.n_generate,
                    n_evaluate=args.n_evaluate,
                    n_select=args.n_select,
                    temperature=args.temperature,
                    propose_max_new=args.propose_max_new,
                    value_max_new=args.value_max_new,
                    batch_size=args.batch_size)
            except Exception as e:
                fout.write(json.dumps({
                    **{k: rec[k] for k in rec if k != "prompt"},
                    "error": str(e)[:200],
                }) + "\n")
                fout.flush()
                continue
            ys = res["ys"]
            validities = [adapter.is_correct(y) for y in ys]
            top1 = bool(validities[0]) if validities else False
            any_correct = any(validities)
            n_top1 += int(top1)
            n_any += int(any_correct)
            total += 1
            fout.write(json.dumps({
                **{k: rec[k] for k in rec if k != "prompt"},
                "ys": ys, "validities": validities,
                "top1_correct": top1, "any_correct": any_correct,
                "history": res["history"],
            }) + "\n")
            fout.flush()
            if (i + 1) % 5 == 0 or i == len(records) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(records) - (i + 1)) / rate
                print(f"  [r{args.shard_rank}] {i+1}/{len(records)} "
                       f"top1={n_top1/max(total,1):.3f} "
                       f"any={n_any/max(total,1):.3f} "
                       f"({elapsed/60:.1f}m, eta={eta/60:.1f}m)", flush=True)

    print(f"  [r{args.shard_rank}] done. top1={n_top1}/{total}, "
           f"any={n_any}/{total}", flush=True)


if __name__ == "__main__":
    main()
