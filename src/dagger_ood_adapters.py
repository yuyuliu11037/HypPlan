"""Task adapters for DAgger Stage-2 in-domain training on PQ / BW / GC.

Each adapter exposes the API the generic OOD rollout + trainer call:
  - parse_problem(rec)             — build internal Problem state from a test/train record
  - initial_state                  — frozenset / tuple representing root
  - winning_steps(state)           — list of step-actions that move toward goal
  - validate_apply(state, action)  — (legal, new_state)
  - is_solved(state) / is_terminal(state)
  - render_state(state, history)   — text for Stage-1 head's z input
  - parse_step(generation_text, state, history) -> (action, raw_step_text) | None
  - format_step_text(state_before, action, state_after, step_num, max_steps) — gold CE target
  - make_prompt(tokenizer)         — (prompt_text, add_special)
  - step_priming_prefix(step_num)  — text appended after prompt to seed generation
  - BOUNDARY_RE / TERMINAL_RE      — regex patterns for boundary detection in generated text

A single uniform output format is used across tasks for DAgger CE supervision:
  Step 1: <task-specific content>
  Step 2: <task-specific content>
  ...
  Step K: <task-specific content>. Answer: <final>

The adapter parses the model's emitted "Step N: ..." into a task-specific Action
internally. This way the rollout/trainer code stays task-agnostic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


# Shared boundary / answer regex (unified output format).
BOUNDARY_RE = re.compile(r"\nStep \d+:")
ANSWER_RE = re.compile(r"Answer\s*:")


# ---------- ProntoQA ----------


_PQ_HF_CACHE = None  # id -> raw_logic_programs


def _pq_load_hf():
    global _PQ_HF_CACHE
    if _PQ_HF_CACHE is not None:
        return _PQ_HF_CACHE
    from datasets import load_dataset
    ds = load_dataset("renma/ProntoQA", split="validation")
    _PQ_HF_CACHE = {rec["id"]: rec["raw_logic_programs"] for rec in ds}
    return _PQ_HF_CACHE


class ProntoQAAdapter:
    name = "pq"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_pronto import parse_problem
        self.rec = rec
        # Look up raw_logic_programs from HF by id (same source the head was
        # trained from). If id not found, raise.
        hf = _pq_load_hf()
        rid = rec["id"]
        if rid not in hf:
            raise KeyError(f"ProntoQA id {rid} not found in HF dataset")
        self.problem = parse_problem(hf[rid])
        self.entity = self.problem.entity
        self.gold_label = rec.get("answer_label", None)
        self._tree = None

    @property
    def initial_state(self):
        return self.problem.facts

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_pronto import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=2000)
        return self._tree

    def winning_steps(self, state):
        """Return list of (rule, new_state) tuples representing gold next steps.
        Compares v_value: child must have smaller v_value than current."""
        from src.oracle_pronto import forward_apply
        tree = self._tree_lazy()
        node_id = None
        for n in tree.nodes:
            if n.state == state:
                node_id = n.node_id
                break
        if node_id is None:
            return []
        node = tree.nodes[node_id]
        if node.v_value <= 0:
            return []
        winners = []
        for rule in self.problem.rules:
            new_state = forward_apply(state, rule)
            if new_state is None:
                continue
            # Look up child node
            for cid in node.children:
                if tree.nodes[cid].state == new_state:
                    if (tree.nodes[cid].v_value >= 0
                        and tree.nodes[cid].v_value < node.v_value):
                        winners.append((rule, new_state))
                    break
        return winners

    def validate_apply(self, state, action):
        """action = (rule, new_state). Return (legal, new_state)."""
        from src.oracle_pronto import forward_apply
        rule, _ = action
        new_state = forward_apply(state, rule)
        if new_state is None:
            return False, state
        return True, new_state

    def is_solved(self, state):
        from src.oracle_pronto import decidable
        return decidable(state, self.problem.query)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_pronto import render_state as render
        return render(self.problem, state)

    def _action_text(self, rule, new_state):
        """Render the derived fact in NL form for the model output."""
        verb = "is" if rule.conclusion_val else "is not"
        return (f"since {self.entity} is {rule.premise_pred}, "
                f"{self.entity} {verb} {rule.conclusion_pred}")

    def parse_step(self, step_body: str, state, history):
        """Parse a single emitted step (without the "Step N:" prefix).
        Match against self.problem.rules: which rule produced this fact?"""
        from src.oracle_pronto import forward_apply
        s = step_body.strip().rstrip(".").lower()
        for rule in self.problem.rules:
            text = self._action_text(rule, None).lower()
            if text in s or s.endswith(text.split(", ")[-1]):
                new_state = forward_apply(state, rule)
                if new_state is not None:
                    return (rule, new_state), step_body.strip()
        return None

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        rule, _ = action
        body = self._action_text(rule, state_after)
        if self.is_solved(state_after):
            ans = self._answer_letter(state_after)
            return f" {body}. Answer: {ans}"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def _answer_letter(self, state):
        qpred, qval = self.problem.query
        for (p, v) in state:
            if p == qpred:
                return "A" if v == qval else "B"
        return "B"

    def make_prompt(self, tokenizer):
        # Render rules + init + query directly from the structured Problem.
        sys = ("You will solve a logical reasoning task. Apply rules step by "
                "step from the initial fact, then output the answer letter. "
                "Output format:\n"
                "  Step 1: since X is P1, X is P2.\n"
                "  Step 2: since X is P2, X is P3.\n"
                "  ...\n"
                "  Step K: since X is Pj, X is Pk. Answer: A   (or B)\n"
                "A=true, B=false. Use ONLY the rules given. One rule per step.")
        # Build NL rules from structured Rule objects.
        rules_lines: list[str] = []
        for rule in self.problem.rules:
            prem = f"{rule.premise_pred}" if rule.premise_val \
                else f"not {rule.premise_pred}"
            conc = f"{rule.conclusion_pred}" if rule.conclusion_val \
                else f"not {rule.conclusion_pred}"
            rules_lines.append(f"If x is {prem}, then x is {conc}.")
        rules_text = "\n".join(rules_lines)
        # Init fact (one fact, the seed of the chain)
        init = next(iter(self.problem.facts))
        init_pred, init_val = init
        init_text = (f"{self.entity} is "
                       f"{'' if init_val else 'not '}{init_pred}")
        qpred, qval = self.problem.query
        q_text = (f"Is {self.entity} "
                    f"{'' if qval else 'not '}{qpred}")
        user = (f"Rules:\n{rules_text}\n\n"
                f"Initial fact: {init_text}.\n"
                f"Question: {q_text}?")
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


# ---------- Blocksworld ----------


class BlocksworldAdapter:
    name = "bw"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_blocksworld import parse_problem
        self.rec = rec
        # Test records expose `prompt`; train records use `question`.
        text = rec.get("prompt") or rec.get("question")
        if text is None:
            raise KeyError("BW record needs `prompt` or `question`")
        self.problem = parse_problem(text)
        self._tree = None

    @property
    def initial_state(self):
        return self.problem.init

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_blocksworld import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=30000)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_blocksworld import (applicable_actions, apply_action,
                                              is_goal)
        tree = self._tree_lazy()
        node_id = None
        for n in tree.nodes:
            if n.state == state:
                node_id = n.node_id
                break
        if node_id is None:
            # State not in tree; on-the-fly fallback: any applicable action.
            return []
        node = tree.nodes[node_id]
        if node.v_value <= 0:
            return []
        winners = []
        for a in applicable_actions(state, self.problem.blocks):
            ns = apply_action(state, a)
            for cid in node.children:
                if tree.nodes[cid].state == ns:
                    if (tree.nodes[cid].v_value >= 0
                        and tree.nodes[cid].v_value < node.v_value):
                        winners.append((a, ns))
                    break
        return winners

    def validate_apply(self, state, action):
        from src.oracle_blocksworld import apply_action
        a, _ = action
        ns = apply_action(state, a)
        if ns == state:
            return False, state
        return True, ns

    def is_solved(self, state):
        from src.oracle_blocksworld import is_goal
        return is_goal(state, self.problem.goal)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_blocksworld import render_state
        return render_state(self.problem, state)

    def _action_text(self, action):
        op, args = action.op, action.args
        if op == "pick-up":
            return f"pick up the {args[0]} block"
        if op == "put-down":
            return f"put down the {args[0]} block"
        if op == "stack":
            return f"stack the {args[0]} block on top of the {args[1]} block"
        if op == "unstack":
            return f"unstack the {args[0]} block from on top of the {args[1]} block"
        return ""

    def parse_step(self, step_body: str, state, history):
        from src.oracle_blocksworld import (applicable_actions, apply_action,
                                              Action)
        s = step_body.strip()
        # Try natural-language patterns first, then PDDL.
        m = re.search(
            r"unstack the (\w+) block from(?: on top of)? the (\w+) block",
            s, re.I)
        if m:
            cand = Action(op="unstack",
                            args=(m.group(1).lower(), m.group(2).lower()))
        else:
            m = re.search(
                r"stack the (\w+) block on(?:to| top of)? the (\w+) block",
                s, re.I)
            if m:
                cand = Action(op="stack",
                                args=(m.group(1).lower(), m.group(2).lower()))
            else:
                m = re.search(r"pick(?:[ -])up the (\w+) block", s, re.I)
                if m:
                    cand = Action(op="pick-up",
                                   args=(m.group(1).lower(),))
                else:
                    m = re.search(r"put(?: down|-down) the (\w+) block",
                                    s, re.I)
                    if m:
                        cand = Action(op="put-down",
                                       args=(m.group(1).lower(),))
                    else:
                        m = re.match(
                            r"^\(?(pick-up|put-down|stack|unstack)"
                            r"\s+(\w+)(?:\s+(\w+))?\)?", s)
                        if m:
                            args = (m.group(2),) if not m.group(3) \
                                else (m.group(2), m.group(3))
                            cand = Action(op=m.group(1), args=args)
                        else:
                            return None
        ns = apply_action(state, cand)
        if ns == state:
            return None
        return (cand, ns), step_body.strip()

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        a, _ = action
        body = self._action_text(a)
        if self.is_solved(state_after):
            return f" {body}. Answer: done"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will produce a sequence of Blocksworld actions to reach "
                "the goal. Each step is one action. Output format:\n"
                "  Step 1: pick up the X block.\n"
                "  Step 2: stack the X block on top of the Y block.\n"
                "  ...\n"
                "  Step K: <action>. Answer: done\n"
                "Use only: pick up X / put down X / stack X on top of Y / "
                "unstack X from on top of Y. Each action must be valid in "
                "the current state.")
        from src.oracle_blocksworld import render_state
        init_text = render_state(self.problem, self.problem.init)
        # Append goal explicitly (render_state already does, but be safe)
        user = (f"{init_text}")
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


# ---------- GraphColor ----------


class GraphColorAdapter:
    name = "gc"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_graphcolor import Problem
        self.rec = rec
        self.problem = Problem(n=rec["n"],
                                edges=tuple(map(tuple, rec["edges"])))
        self._tree = None

    @property
    def initial_state(self):
        return ()  # tuple of (vertex, color)

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_graphcolor import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=8000)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_graphcolor import _next_uncolored, _conflicts
        tree = self._tree_lazy()
        node_id = None
        for n in tree.nodes:
            if n.state == state:
                node_id = n.node_id
                break
        if node_id is None:
            return []
        node = tree.nodes[node_id]
        if node.v_value <= 0:
            return []
        v = _next_uncolored(state, self.problem.n)
        if v is None:
            return []
        winners = []
        for c in ("R", "G", "B"):
            if _conflicts(state, v, c, self.problem.adj()):
                continue
            ns = state + ((v, c),)
            for cid in node.children:
                if tree.nodes[cid].state == ns:
                    if (tree.nodes[cid].v_value >= 0
                        and tree.nodes[cid].v_value < node.v_value):
                        winners.append(((v, c), ns))
                    break
        return winners

    def validate_apply(self, state, action):
        from src.oracle_graphcolor import _conflicts
        (v, c), _ = action
        if _conflicts(state, v, c, self.problem.adj()):
            return False, state
        ns = state + ((v, c),)
        return True, ns

    def is_solved(self, state):
        return len(state) == self.problem.n

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_graphcolor import render_state
        return render_state(self.problem, state)

    def _action_text(self, action):
        (v, c), _ = action
        color = {"R": "red", "G": "green", "B": "blue"}[c]
        return f"V{v} = {color}"

    def parse_step(self, step_body: str, state, history):
        from src.oracle_graphcolor import _conflicts, _next_uncolored
        s = step_body.strip()
        m = re.search(r"V(\d+)\s*[:=]\s*(red|green|blue|R|G|B)\b", s, re.I)
        if not m:
            return None
        v = int(m.group(1))
        c = m.group(2).upper()[0]
        nxt = _next_uncolored(state, self.problem.n)
        if v != nxt:
            return None
        if _conflicts(state, v, c, self.problem.adj()):
            return None
        ns = state + ((v, c),)
        return (((v, c), ns), step_body.strip())

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        (v, c), _ = action
        color = {"R": "red", "G": "green", "B": "blue"}[c]
        body = f"V{v} = {color}"
        if self.is_solved(state_after):
            return f" {body}. Answer: done"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve a graph 3-coloring task. Assign each vertex "
                "one of {red, green, blue} such that adjacent vertices differ. "
                "Output one assignment per step. Output format:\n"
                "  Step 1: V0 = red.\n"
                "  Step 2: V1 = green.\n"
                "  ...\n"
                "  Step K: V{n-1} = <color>. Answer: done\n"
                "Color vertices in order V0, V1, V2, ..., V{n-1}.")
        edges = ", ".join(f"(V{u},V{v})" for u, v in self.problem.edges)
        verts = ", ".join(f"V{i}" for i in range(self.problem.n))
        user = (f"Vertices: {verts}\nEdges: {edges}")
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


# ---------- RuleChain (Group B training source) / Synthlogic (OOD) ----------


class RuleChainAdapter:
    """Adapter for synthetic Horn-clause forward chaining.

    Used both for the Group B training source (`rulechain`) and for the
    harder OOD eval (`synthlogic`). The two share the same primitive; they
    differ only in JSONL data parameters (depth, predicate count,
    pred_prefix). One adapter class, two registry entries."""
    name = "rulechain"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_rulechain import Problem, Rule
        self.rec = rec
        rules = tuple(
            Rule(premises=tuple(sorted(r["premises"])),
                 conclusion=r["conclusion"])
            for r in rec["rules"]
        )
        self.problem = Problem(
            initial_facts=frozenset(rec["initial_facts"]),
            target=rec["target"],
            rules=rules,
        )
        self._tree = None

    @property
    def initial_state(self):
        return self.problem.initial_facts

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_rulechain import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=20000,
                                         max_depth=12)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_rulechain import applicable_rules, apply_rule
        tree = self._tree_lazy()
        node_id = None
        for n in tree.nodes:
            if n.state == state:
                node_id = n.node_id
                break
        if node_id is None:
            return []
        node = tree.nodes[node_id]
        if node.v_value <= 0:
            return []
        winners = []
        for r in applicable_rules(state, self.problem.rules):
            ns = apply_rule(state, r)
            for cid in node.children:
                if tree.nodes[cid].state == ns:
                    if (tree.nodes[cid].v_value >= 0
                        and tree.nodes[cid].v_value < node.v_value):
                        winners.append((r, ns))
                    break
        return winners

    def validate_apply(self, state, action):
        from src.oracle_rulechain import validate_step
        rule, _ = action
        legal, ns = validate_step(state, rule, self.problem)
        return legal, ns

    def is_solved(self, state):
        from src.oracle_rulechain import decidable
        return decidable(state, self.problem.target)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_rulechain import render_state
        return render_state(self.problem, state)

    def _action_text(self, rule):
        from src.oracle_rulechain import format_step_text
        return format_step_text(rule)

    def parse_step(self, step_body: str, state, history):
        from src.oracle_rulechain import (
            apply_rule, applicable_rules, parse_step,
        )
        rule = parse_step(step_body, self.problem)
        if rule is None:
            return None
        if rule not in self.problem.rules:
            return None
        if rule.conclusion in state:
            return None
        if not all(p in state for p in rule.premises):
            return None
        ns = apply_rule(state, rule)
        return (rule, ns), step_body.strip()

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        rule, _ = action
        body = self._action_text(rule)
        if self.is_solved(state_after):
            return f" {body}. Answer: {self.problem.target} is derived"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve a forward-chaining derivation task. Apply one "
                "rule per step until the target predicate is derived. Output "
                "format:\n"
                "  Step 1: apply rule: if A and B, then C.\n"
                "  Step 2: apply rule: if C, then D.\n"
                "  ...\n"
                "  Step K: apply rule: if ..., then <target>. Answer: <target> "
                "is derived\n"
                "Use ONLY the rules given. Each step must apply exactly one "
                "rule whose premises are already known.")
        rules_text = "\n".join(
            f"- {r.render()}" for r in self.problem.rules
        )
        init = ", ".join(sorted(self.problem.initial_facts))
        user = (f"Rules:\n{rules_text}\n\n"
                f"Initial facts: {init}\n"
                f"Goal: derive {self.problem.target}")
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


class SynthlogicAdapter(RuleChainAdapter):
    """OOD eval-only synthlogic. Identical primitive to RuleChainAdapter;
    distinct registry entry so eval drivers and configs can reference
    `synthlogic` as its own task."""
    name = "synthlogic"


# ---------- CLUTRR-like (kinship reasoning, OOD) ----------


class CLUTRRAdapter:
    name = "clutrr"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_clutrr import Problem
        self.rec = rec
        self.problem = Problem(
            entities=tuple(rec["entities"]),
            edges=tuple((i, rel, j) for (i, rel, j) in rec["edges"]),
            query=tuple(rec["query"]),
            answer=rec["answer"],
            chain=tuple(rec["chain"]),
        )
        self._tree = None

    @property
    def initial_state(self):
        return ()  # empty composition prefix

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_clutrr import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=200)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_clutrr import winning_steps as ws, validate_step
        tree = self._tree_lazy()
        d = len(state)
        if d >= len(self.problem.chain):
            return []
        wins = ws(state, self.problem)
        out = []
        for (h, rel) in wins:
            legal, ns = validate_step(state, h, rel, self.problem)
            if legal:
                out.append(((h, rel), ns))
        return out

    def validate_apply(self, state, action):
        from src.oracle_clutrr import validate_step
        (h, rel), _ = action
        legal, ns = validate_step(state, h, rel, self.problem)
        return legal, ns

    def is_solved(self, state):
        from src.oracle_clutrr import is_solved
        return is_solved(state, self.problem)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_clutrr import render_state
        return render_state(self.problem, state)

    def _action_text(self, action, state_after):
        # state_after's last element is the new derived relation.
        head = self.problem.entities[self.problem.query[0]]
        d = len(state_after) - 1
        intermediate = self.problem.entities[d + 1]
        derived_rel = state_after[-1]
        return f"{head} is the {derived_rel} of {intermediate}"

    def parse_step(self, step_body: str, state, history):
        from src.oracle_clutrr import parse_step, validate_step
        out = parse_step(step_body, self.problem, len(state))
        if out is None:
            return None
        h, rel = out
        legal, ns = validate_step(state, h, rel, self.problem)
        if not legal:
            return None
        return ((h, rel), ns), step_body.strip()

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        body = self._action_text(action, state_after)
        if self.is_solved(state_after):
            tail = self.problem.entities[self.problem.query[1]]
            return (f" {body}. Answer: "
                    f"{self.problem.entities[self.problem.query[0]]} is the "
                    f"{self.problem.answer} of {tail}")
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve a kinship-relation composition task. Read the "
                "story, then derive the relation between the queried head and "
                "tail entities by composing one kinship hop per step. Output "
                "format:\n"
                "  Step 1: <head> is the <relation> of <intermediate1>.\n"
                "  Step 2: <head> is the <relation> of <intermediate2>.\n"
                "  ...\n"
                "  Step K: <head> is the <derived_relation> of <tail>. "
                "Answer: <head> is the <relation> of <tail>")
        story_lines = [
            f"{self.problem.entities[i]} is the {rel} of "
            f"{self.problem.entities[j]}."
            for (i, rel, j) in self.problem.edges
        ]
        head = self.problem.entities[self.problem.query[0]]
        tail = self.problem.entities[self.problem.query[1]]
        user = (
            "\n".join(story_lines)
            + f"\n\nQuestion: How is {head} related to {tail}?"
        )
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


# ---------- Linear Equations (Group A OOD #1) ----------


class LinearEqAdapter:
    name = "lineq"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_lineq import Problem, State
        self.rec = rec
        init = rec["initial"]
        self.problem = Problem(
            initial=State(
                lhs_x=tuple(init["lhs_x"]),
                lhs_c=tuple(init["lhs_c"]),
                rhs_x=tuple(init["rhs_x"]),
                rhs_c=tuple(init["rhs_c"]),
            ),
            solution=int(rec["solution"]),
        )
        self._tree = None

    @property
    def initial_state(self):
        return self.problem.initial

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_lineq import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=1000,
                                         max_depth=10)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_lineq import winning_steps as ws, apply_op
        wins = ws(state, self.problem)
        out = []
        for op in wins:
            ns = apply_op(state, op)
            if ns is not None:
                out.append((op, ns))
        return out

    def validate_apply(self, state, action):
        from src.oracle_lineq import validate_step
        op, _ = action
        legal, ns = validate_step(state, op)
        return legal, ns

    def is_solved(self, state):
        from src.oracle_lineq import is_solved
        return is_solved(state, self.problem.solution)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_lineq import render_state
        return render_state(self.problem, state)

    def _action_text(self, op, state_after):
        from src.oracle_lineq import format_step_text
        return format_step_text(op, state_after)

    def parse_step(self, step_body: str, state, history):
        from src.oracle_lineq import parse_step, validate_step
        op = parse_step(step_body)
        if op is None:
            return None
        legal, ns = validate_step(state, op)
        if not legal:
            return None
        return ((op, ns), step_body.strip())

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        op, _ = action
        body = self._action_text(op, state_after)
        if self.is_solved(state_after):
            return f" {body}. Answer: x = {self.problem.solution}"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve a single-variable linear equation. Apply one "
                "canonical operation per step until the equation is in "
                "x = K form. Allowed operations:\n"
                "  - combine like x-terms on the left/right\n"
                "  - combine constants on the left/right\n"
                "  - subtract A*x from both sides\n"
                "  - subtract B from both sides\n"
                "  - divide both sides by C\n"
                "Output format:\n"
                "  Step 1: <operation> -> <new equation>.\n"
                "  Step 2: <operation> -> <new equation>.\n"
                "  ...\n"
                "  Step K: divide both sides by C -> x = N. Answer: x = N\n"
                "Solve in canonical order: combine first, then move x, then "
                "move constants, then divide.")
        from src.oracle_lineq import format_question
        user = format_question(self.problem)
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


# ---------- Number-path (Group A OOD #1) ----------


class NumPathAdapter:
    name = "numpath"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_numpath import Op, Problem
        self.rec = rec
        ops = tuple(Op(o["kind"], int(o["const"])) for o in rec["ops"])
        self.problem = Problem(
            start=int(rec["start"]),
            target=int(rec["target"]),
            ops=ops,
            max_value=int(rec.get("max_value", 999)),
        )
        self._tree = None

    @property
    def initial_state(self):
        return self.problem.start

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_numpath import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=2000,
                                         max_depth=12)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_numpath import (
            apply_op, winning_steps as ws,
        )
        wins = ws(state, self.problem)
        out = []
        for op in wins:
            ns = apply_op(state, op, self.problem)
            if ns is not None:
                out.append((op, ns))
        return out

    def validate_apply(self, state, action):
        from src.oracle_numpath import validate_step
        op, _ = action
        legal, ns = validate_step(state, op, self.problem)
        return legal, ns

    def is_solved(self, state):
        from src.oracle_numpath import is_solved
        return is_solved(state, self.problem.target)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_numpath import render_state
        return render_state(self.problem, state)

    def _action_text(self, state_before, op, state_after):
        from src.oracle_numpath import format_step_text
        return format_step_text(state_before, op, state_after, 0)

    def parse_step(self, step_body: str, state, history):
        from src.oracle_numpath import parse_step
        out = parse_step(step_body, self.problem, state)
        if out is None:
            return None
        op, ns = out
        return ((op, ns), step_body.strip())

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        op, _ = action
        body = self._action_text(state_before, op, state_after)
        if self.is_solved(state_after):
            return f" {body}. Answer: {self.problem.target}"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve a number-path puzzle. Apply operations from "
                "the given set to transform the start number into the "
                "target. Each step uses one operation. Output format:\n"
                "  Step 1: a op b = r\n"
                "  Step 2: r op b = s\n"
                "  ...\n"
                "  Answer: <target>\n"
                "Subtraction must give a non-negative result. Division "
                "must be exact. The current value is the LHS of each step.")
        from src.oracle_numpath import format_question
        user = format_question(self.problem)
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


# ---------- ProofWriter (CWA depth-3, NL deductive proof) ----------


class ProofWriterAdapter:
    name = "proofwriter"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = ANSWER_RE

    def __init__(self, rec: dict):
        from src.oracle_proofwriter import Problem
        self.rec = rec
        triple_texts = {
            tuple(k): v for (k, v) in rec.get("triple_texts", [])
        }
        self.problem = Problem(
            theory_text=rec["theory_text"],
            initial_facts=tuple(tuple(t) for t in rec["initial_facts"]),
            rule_texts=dict(rec["rule_texts"]),
            rules_struct=dict(rec["rules_struct"]),
            triple_texts=triple_texts,
            target=tuple(rec["target"]),
            target_text=rec["target_text"],
            answer=bool(rec["answer"]),
            proof_chain=tuple({
                "rule_id": s["rule_id"],
                "intermediate": tuple(s["intermediate"]),
                "intermediate_text": s["intermediate_text"],
            } for s in rec["proof_chain"]),
        )
        self._tree = None

    @property
    def initial_state(self):
        return frozenset(self.problem.initial_facts)

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_proofwriter import enumerate_tree
            self._tree = enumerate_tree(self.problem)
        return self._tree

    def winning_steps(self, state):
        from src.oracle_proofwriter import winning_steps as ws
        wins = ws(state, self.problem)
        out = []
        for step in wins:
            ns = state | {step["intermediate"]}
            out.append((step, ns))
        return out

    def validate_apply(self, state, action):
        from src.oracle_proofwriter import validate_step
        step, _ = action
        legal, ns = validate_step(state, step, self.problem)
        return legal, ns

    def is_solved(self, state):
        from src.oracle_proofwriter import is_solved
        return is_solved(state, self.problem.target, self.problem.answer,
                          self.problem.proof_chain)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_proofwriter import render_state
        return render_state(self.problem, state)

    def _action_text(self, step):
        from src.oracle_proofwriter import format_step_text
        return format_step_text(step)

    def parse_step(self, step_body: str, state, history):
        from src.oracle_proofwriter import parse_step
        step = parse_step(step_body, self.problem, state)
        if step is None:
            return None
        ns = state | {step["intermediate"]}
        return ((step, ns), step_body.strip())

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        step, _ = action
        body = self._action_text(step)
        if self.is_solved(state_after):
            ans = "True" if self.problem.answer else "False"
            return f" {body}. Answer: {ans}"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve a deductive reasoning task. Read the theory "
                "and apply rules step by step until you can decide whether "
                "the question's statement is true or false. Output format:\n"
                "  Step 1: apply ruleN: <derived fact in NL>.\n"
                "  Step 2: apply ruleM: <derived fact>.\n"
                "  ...\n"
                "  Step K: apply ruleK: <final derived fact>. Answer: True\n"
                "Output 'Answer: True' if the question follows from the "
                "theory, 'Answer: False' otherwise. Use only the rules in "
                "the theory. Steps may be omitted if the answer is "
                "directly stated in the initial facts.")
        from src.oracle_proofwriter import format_question
        user = format_question(self.problem)
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        return f"Step {step_num}:"


class NQueensAdapter:
    """Adapter for N-Queens with optional pre-placed queens.

    Record schema: {"id", "N", "k", "prefix": list[int], "gold_extension": list[int]}
    State = tuple of column placements (1-indexed) for rows 1..len(state).
    Action = column number (int) chosen for the next row.
    """
    name = "nqueens"
    BOUNDARY_RE = BOUNDARY_RE
    TERMINAL_RE = re.compile(r"Solution\s*[:=]?\s*\[", re.IGNORECASE)

    def __init__(self, rec: dict):
        from src.oracle_nqueens import Problem
        self.rec = rec
        self.problem = Problem(N=int(rec["N"]),
                                prefix=tuple(rec.get("prefix", [])))
        self._tree = None

    @property
    def initial_state(self):
        return self.problem.initial_state()

    def _tree_lazy(self):
        if self._tree is None:
            from src.oracle_nqueens import enumerate_tree
            self._tree = enumerate_tree(self.problem, max_nodes=4000)
        return self._tree

    def winning_steps(self, state):
        """Return list of (col_int, new_state) pairs. The framework expects
        each item to be (action_data, new_state); for N-Queens the
        action_data is just the column int placed at the next row."""
        from src.oracle_nqueens import winning_steps as ws
        out = []
        for col, new_state in ws(state, self.problem):
            out.append((col, new_state))
        return out

    def validate_apply(self, state, action):
        """`action` shape: (col_int, new_state). Returns (ok, new_state)."""
        from src.oracle_nqueens import validate_step
        col, _ = action
        ok, ns = validate_step(state, col, self.problem)
        return ok, ns

    def is_solved(self, state):
        from src.oracle_nqueens import is_solved
        return is_solved(state, self.problem)

    def is_terminal(self, state):
        return self.is_solved(state)

    def render_state(self, state, history):
        from src.oracle_nqueens import render_state
        return render_state(self.problem, state)

    def _action_text(self, action):
        col, _ = action
        next_row = "?"  # not used outside step formatting
        return f"col {col}"

    def parse_step(self, step_body: str, state, history):
        """Returns ((col_int, new_state), step_body) so the framework
        unpacks `action = (col_int, new_state)` consistently with
        winning_steps."""
        from src.oracle_nqueens import parse_step as ps
        col = ps(step_body, self.problem, state)
        if col is None:
            return None
        ns = state + (col,)
        return ((col, ns), step_body.strip())

    def format_step_text(self, state_before, action, state_after,
                          step_num, max_steps):
        col, _ = action
        next_row = len(state_before) + 1
        body = f"Place queen in row {next_row} at column {col}"
        if self.is_solved(state_after):
            sol_str = ", ".join(str(c) for c in state_after)
            return f" {body}. Solution: [{sol_str}]"
        tail = ""
        next_n = step_num + 1
        if next_n <= max_steps:
            tail = f"\nStep {next_n}:"
        return f" {body}." + tail

    def make_prompt(self, tokenizer):
        sys = ("You will solve an N-Queens puzzle. Place N queens on an "
                "N x N board, one per row, so that no two queens share "
                "the same column or diagonal. At each step, place a queen "
                "in the next empty row at a valid column (1-indexed). "
                "Output format:\n"
                "  Step 1: Place queen in row 1 at column c1.\n"
                "  Step 2: Place queen in row 2 at column c2.\n"
                "  ...\n"
                "  Step N: Place queen in row N at column cN. "
                "Solution: [c1, c2, ..., cN]")
        N = self.problem.N
        if self.problem.prefix:
            pre = "Already-placed queens:\n" + "\n".join(
                f"  row {r} col {c}"
                for r, c in enumerate(self.problem.prefix, 1))
        else:
            pre = "No queens placed yet."
        user = f"Board size: {N}x{N}\n{pre}"
        msgs = [{"role": "system", "content": sys},
                 {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return text, False

    def step_priming_prefix(self, step_num):
        """Step number = ACTION index (1-indexed), NOT row number.

        Training convention (see train_stage2_dagger_ood.py format_step_text
        call): for prefixed problems, "Step 1" is the first model action,
        which places at row len(prefix)+1. The user-message of make_prompt
        already conveys pre-placed queens; do not render them in the
        assistant priming, which would create a format the model never
        saw during training."""
        return f"Step {step_num}:"


ADAPTERS = {
    "pq": ProntoQAAdapter, "bw": BlocksworldAdapter,
    "gc": GraphColorAdapter,
    "rulechain": RuleChainAdapter,
    "synthlogic": SynthlogicAdapter,
    "clutrr": CLUTRRAdapter,
    "lineq": LinearEqAdapter,
    "numpath": NumPathAdapter,
    "proofwriter": ProofWriterAdapter,
    "nqueens": NQueensAdapter,
}
