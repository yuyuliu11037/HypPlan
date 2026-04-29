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

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        out: list[str] = []
        seen: set = set()
        for ln in gen.splitlines():
            ln = ln.strip().strip("()")
            ln = re.sub(r"^[\s\-\*\d\.]+", "", ln)
            cand = None
            if m := re.search(
                r"unstack the (\w+) block from(?: on top of| from)? the (\w+) block",
                ln, re.I):
                cand = f"unstack {m.group(1).lower()} {m.group(2).lower()}"
            elif m := re.search(
                r"stack the (\w+) block on(?:to| top of)? the (\w+) block",
                ln, re.I):
                cand = f"stack {m.group(1).lower()} {m.group(2).lower()}"
            elif m := re.search(r"pick(?:[ -])up the (\w+) block", ln, re.I):
                cand = f"pick-up {m.group(1).lower()}"
            elif m := re.search(r"put(?: down|-down)? the (\w+) block",
                                 ln, re.I):
                cand = f"put-down {m.group(1).lower()}"
            else:
                m = re.match(
                    r"^(pick-up|put-down|stack|unstack)\s+(\w+)(?:\s+(\w+))?\b",
                    ln)
                if m:
                    op = m.group(1)
                    if m.group(3):
                        cand = f"{op} {m.group(2)} {m.group(3)}"
                    else:
                        cand = f"{op} {m.group(2)}"
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

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
    "Propose 3 different candidate color assignments (red/green/blue) for "
    "the NEXT uncolored vertex in this 3-coloring problem. Each line must "
    "be of the form 'V<i> = <color>' for the SAME vertex i (the next "
    "uncolored one). Output exactly 3 lines, no commentary.\n\n"
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

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        nxt = self._next_uncolored(partial)
        out: list[str] = []
        seen: set = set()
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
                cand = f"V{v} = {color_full}"
                if cand not in seen:
                    seen.add(cand)
                    out.append(cand)
        return out

    def is_terminal(self, partial: str) -> bool:
        c = self._coloring_so_far(partial)
        return len(c) >= self.problem.n

    def is_correct(self, partial: str) -> bool:
        from src.oracle_graphcolor import score_coloring
        c = self._coloring_so_far(partial)
        return score_coloring(self.problem, c)


# ---------- ProofWriter adapter ----------

_PW_PROPOSE_HEADER = (
    "You will derive new facts step by step from the rules, then conclude "
    "with the answer (True or False). Each step uses one rule and one or "
    "more previously known facts.\n"
    "Propose 3 different sensible next-line candidates for the derivation "
    "below. Output exactly 3 lines (no numbering, no commentary). Each "
    "line must be either\n"
    "  Step N: apply ruleM: <derived fact in NL>.\n"
    "or\n"
    "  Answer: True\n"
    "or\n"
    "  Answer: False\n"
    "If the question's statement is already in the initial facts (or "
    "trivially negated under the closed-world assumption), output "
    "'Answer: True' or 'Answer: False' directly.\n\n"
)

_PW_VALUE_HEADER = (
    "Given the theory (initial facts + rules) and the target question, "
    "decide whether the partial derivation is on track to arrive at the "
    "correct True/False conclusion. Reply with EXACTLY one of: sure / "
    "likely / impossible. No other text.\n\n"
)


class ProofWriterAdapter(Adapter):
    name = "proofwriter"
    max_depth = 12

    def __init__(self, rec: dict) -> None:
        super().__init__(rec)
        self.theory = rec["theory_text"]
        self.target = rec["target_text"]
        self.gold_answer = bool(rec["answer"])

    def _common(self) -> str:
        return (f"Theory:\n{self.theory}\n\n"
                f"Question: is the following true? "
                f"{self.target}\n\n")

    def propose_prompt(self, partial: str) -> str:
        body = self._common()
        n_so_far = len(re.findall(r"^Step\s+\d+:", partial, re.MULTILINE))
        next_n = n_so_far + 1
        if partial:
            body += "Derivation so far:\n" + partial
        else:
            body += "Derivation so far: (none)\n"
        body += f"\nStep {next_n}:"
        return _PW_PROPOSE_HEADER + body

    def value_prompt(self, partial: str) -> str:
        body = self._common()
        body += "Partial derivation:\n" + (partial if partial else "(none)\n")
        body += "\nEvaluation:"
        return _PW_VALUE_HEADER + body

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        out: list[str] = []
        for ln in gen.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r"^[\-\*\d\.]+\s*", "", ln)
            m_ans = re.match(r"^Answer:\s*(True|False)\b", ln, re.I)
            if m_ans:
                out.append(f"Answer: "
                           f"{m_ans.group(1).capitalize()}")
                continue
            if re.match(r"^Step\s+\d+:\s*apply\s+rule\d+", ln, re.I):
                out.append(ln.rstrip())
        return out

    def is_terminal(self, partial: str) -> bool:
        return bool(re.search(r"Answer:\s*(True|False)\b", partial,
                                re.IGNORECASE))

    def is_correct(self, partial: str) -> bool:
        m = re.search(r"Answer:\s*(True|False)\b", partial, re.IGNORECASE)
        if not m:
            return False
        return (m.group(1).lower() == "true") == self.gold_answer


# ---------- N-Queens adapter ----------

_NQ_PROPOSE_HEADER = (
    "Place N queens on an N x N board, one per row, so that no two queens "
    "share the same column or diagonal.\n"
    "Propose 3 different sensible next-line candidates for the partial "
    "placement below. Output exactly 3 lines (no numbering, no commentary). "
    "Each line must be either\n"
    "  Step k: Place queen in row k at column c.\n"
    "or, when all rows are placed,\n"
    "  Solution: [c1, c2, ..., cN]\n\n"
)

_NQ_VALUE_HEADER = (
    "Given the board size (N) and the partial placement of queens, decide "
    "whether continuing this placement will yield a valid full N-queens "
    "solution. Reply with EXACTLY one of: sure / likely / impossible. "
    "No other text.\n\n"
)


class NQueensAdapter(Adapter):
    name = "nqueens"
    max_depth = 16

    def __init__(self, rec: dict) -> None:
        super().__init__(rec)
        from src.oracle_nqueens import Problem
        self.N = int(rec["N"])
        self.problem = Problem(N=self.N,
                                prefix=tuple(rec.get("prefix", [])))

    def init_partial(self) -> str:
        # Pre-render the pre-placed prefix as Step lines so the model
        # continues from row k+1.
        if not self.problem.prefix:
            return ""
        lines = []
        for r, c in enumerate(self.problem.prefix, 1):
            lines.append(f"Step {r}: Place queen in row {r} at column {c}.")
        return "\n".join(lines)

    def _common(self) -> str:
        if self.problem.prefix:
            pre = "\nPre-placed queens (rows 1.." + str(len(self.problem.prefix)) + "):\n"
            for r, c in enumerate(self.problem.prefix, 1):
                pre += f"  row {r} col {c}\n"
        else:
            pre = "\nNo queens placed yet.\n"
        return f"Board size N = {self.N}.{pre}\n"

    def _placed_columns(self, partial: str) -> list[int]:
        cols: list[int] = []
        for ln in partial.splitlines():
            m = re.search(r"row\s+\d+\s+at\s+column\s+(\d+)", ln,
                          re.IGNORECASE)
            if m:
                cols.append(int(m.group(1)))
        return cols

    def propose_prompt(self, partial: str) -> str:
        body = self._common()
        n_so_far = len(self._placed_columns(partial))
        next_n = n_so_far + 1
        if partial:
            body += "Placements so far:\n" + partial
        else:
            body += "Placements so far: (none)\n"
        if next_n <= self.N:
            body += f"\nStep {next_n}:"
        else:
            body += "\nSolution:"
        return _NQ_PROPOSE_HEADER + body

    def value_prompt(self, partial: str) -> str:
        body = self._common()
        body += "Partial placement:\n" + (partial if partial else "(none)\n")
        body += "\nEvaluation:"
        return _NQ_VALUE_HEADER + body

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        """Lenient step extractor for N-Queens. Accepts:
          - 'Step N: Place queen in row N at column C.'
          - 'Step N:' followed by anything (loose; column might be inferred)
          - bare 'Place queen in row N at column C.' (when prompt primed Step N:)
          - 'Solution: [...]' (terminal)
        """
        out: list[str] = []
        n_so_far = len(self._placed_columns(partial))
        next_n = n_so_far + 1
        for ln in gen.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r"^[\-\*\d\.]+\s*", "", ln)
            if re.match(r"^Solution\s*[:=]?\s*\[", ln, re.IGNORECASE):
                out.append(ln.rstrip())
                continue
            if re.match(r"^Step\s+\d+:\s*Place\s+queen", ln, re.IGNORECASE):
                out.append(ln.rstrip())
                continue
            # Bare "Place queen in row N at column C." — model continued
            # the priming "Step N:" without re-emitting the prefix.
            if re.match(r"^Place\s+queen\s+in\s+row\s+\d+\s+at\s+column\s+\d+",
                         ln, re.IGNORECASE):
                out.append(f"Step {next_n}: {ln.rstrip()}")
                continue
            # Bare "row N col C" or "column C" if it has digits
            m = re.match(r"^(?:row\s+\d+\s+(?:at\s+)?)?col(?:umn)?\s*[:=]?\s*(\d+)",
                          ln, re.IGNORECASE)
            if m:
                c = int(m.group(1))
                out.append(f"Step {next_n}: Place queen in row {next_n} "
                           f"at column {c}.")
                continue
        return out

    def is_terminal(self, partial: str) -> bool:
        if re.search(r"Solution\s*[:=]?\s*\[", partial, re.IGNORECASE):
            return True
        # Also terminal if N rows placed
        return len(self._placed_columns(partial)) >= self.N

    def is_correct(self, partial: str) -> bool:
        from src.oracle_nqueens import parse_solution, score_solution
        sol = parse_solution(partial)
        if sol is None:
            # Fall back: extract column list directly from Step lines
            cols = self._placed_columns(partial)
            if len(cols) == self.N:
                sol = cols
        if sol is None or len(sol) != self.N:
            return False
        # Prefix must be preserved
        if list(sol[: len(self.problem.prefix)]) != list(self.problem.prefix):
            return False
        return score_solution(self.N, sol)


# ---------- CLUTRR adapter ----------

_CLUTRR_PROPOSE_HEADER = (
    "You will solve a kinship-composition puzzle. Given a list of family "
    "facts, derive how the queried head is related to the queried tail by "
    "composing facts step by step.\n"
    "Propose 3 different sensible next-line candidates for the derivation "
    "below. Output exactly 3 lines (no numbering, no commentary). Each "
    "line must be either\n"
    "  Step N: <Head> is the <relation> of <Person>.\n"
    "or, when ready,\n"
    "  Answer: <Head> is the <kinship-relation> of <Tail>.\n\n"
)

_CLUTRR_VALUE_HEADER = (
    "Given the family facts and the partial kinship derivation, decide "
    "whether continuing it will arrive at the correct relation between "
    "the queried head and tail. Reply with EXACTLY one of: sure / likely "
    "/ impossible. No other text.\n\n"
)


class CLUTRRAdapter(Adapter):
    name = "clutrr"
    max_depth = 8

    def __init__(self, rec: dict) -> None:
        super().__init__(rec)
        self.entities = rec["entities"]
        self.edges = rec["edges"]
        self.query = rec["query"]
        self.gold_answer = rec["answer"]
        # Pre-render the facts + question.
        head_idx, tail_idx = self.query
        self.head_name = self.entities[head_idx]
        self.tail_name = self.entities[tail_idx]

    def _facts_text(self) -> str:
        lines = []
        for src, rel, dst in self.edges:
            lines.append(f"{self.entities[src]} is the {rel} of "
                         f"{self.entities[dst]}.")
        return "\n".join(lines)

    def _common(self) -> str:
        return (f"Family facts:\n{self._facts_text()}\n\n"
                f"Question: How is {self.head_name} related to "
                f"{self.tail_name}?\n\n")

    def propose_prompt(self, partial: str) -> str:
        body = self._common()
        n_so_far = len(re.findall(r"^Step\s+\d+:", partial, re.MULTILINE))
        next_n = n_so_far + 1
        if partial:
            body += "Derivation so far:\n" + partial
        else:
            body += "Derivation so far: (none)\n"
        body += f"\nStep {next_n}:"
        return _CLUTRR_PROPOSE_HEADER + body

    def value_prompt(self, partial: str) -> str:
        body = self._common()
        body += "Partial derivation:\n" + (partial if partial else "(none)\n")
        body += "\nEvaluation:"
        return _CLUTRR_VALUE_HEADER + body

    def extract_steps(self, gen: str, partial: str) -> list[str]:
        out: list[str] = []
        for ln in gen.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r"^[\-\*\d\.]+\s*", "", ln)
            if re.match(r"^Answer\s*[:=]?\s*", ln, re.IGNORECASE):
                out.append(ln.rstrip())
                continue
            if re.match(r"^Step\s+\d+:\s*\w+\s+is\s+the\s+\w",
                         ln, re.IGNORECASE):
                out.append(ln.rstrip())
        return out

    def is_terminal(self, partial: str) -> bool:
        return bool(re.search(r"^Answer\s*[:=]?\s*", partial,
                                re.IGNORECASE | re.MULTILINE))

    def is_correct(self, partial: str) -> bool:
        from src.oracle_clutrr import parse_answer, score_answer
        pred = parse_answer(partial)
        return score_answer(pred, self.gold_answer)


ADAPTERS = {"pq": ProntoQAAdapter, "bw": BlocksworldAdapter,
             "gc": GraphColorAdapter,
             "proofwriter": ProofWriterAdapter,
             "nqueens": NQueensAdapter,
             "clutrr": CLUTRRAdapter}


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
                for step in adapter.extract_steps(text, y):
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

    out_path = Path(args.output)
    if args.shard_world > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}_shard{args.shard_rank}{out_path.suffix}"
        )
    args.output = str(out_path)
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
