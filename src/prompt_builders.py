"""Prompt builders for Game-of-24 DAgger pipeline.

Each builder produces the prompt text used by rollout and training. Returns
`(text, add_special_tokens)`:
- `text` is the full prompt ending right before the model's first-step
  continuation (i.e., ends with "Step 1:").
- `add_special_tokens` tells the tokenizer whether to prepend BOS. For raw
  text prompts we pass True. For chat-template prompts, the template already
  includes the BOS/special tokens, so we pass False to avoid duplicating.

Rollout and training MUST use the same prompt at any given boundary, or the
LoRA would train on one distribution and be tested on another. Both code
paths import from here.
"""
from __future__ import annotations

from typing import Callable, Tuple


INSTRUCTION_24 = (
    "Use the four given numbers and basic arithmetic operations "
    "(+, -, *, /) to obtain 24. Each number must be used exactly once."
)


def _apply_chat_template_no_think(tokenizer, msgs):
    """Wrapper around `apply_chat_template` that passes `enable_thinking=False`
    when the tokenizer supports it (Qwen3 family). Silently falls back for
    older tokenizers (Qwen2.5 and earlier) that don't accept the kwarg.
    """
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def sft_prompt_24(tokenizer, problem: str) -> Tuple[str, bool]:
    """Raw completion prompt matching our SFT training distribution."""
    nums = problem.replace(",", " ")
    text = f"{INSTRUCTION_24}\n\nProblem: {nums}\nStep 1:"
    return text, True


# --- Qwen / chat-template few-shot prompt (Game-24) ---

FEWSHOT_SYSTEM_24 = (
    "You are a careful arithmetic solver. Use the four given numbers and "
    "basic arithmetic operations (+, -, *, /) to obtain 24. Each number "
    "must be used exactly once. Respond with exactly three lines in the "
    "format shown in the examples, ending with 'Answer: 24' on the last "
    "step. Do not add any other text."
)

FEWSHOT_EXAMPLES_24 = [
    (
        "Problem: 4 4 6 8",
        "Step 1: 4 + 8 = 12. Remaining: 4 6 12\n"
        "Step 2: 6 - 4 = 2. Remaining: 2 12\n"
        "Step 3: 2 * 12 = 24. Answer: 24",
    ),
    (
        "Problem: 2 9 10 12",
        "Step 1: 12 * 2 = 24. Remaining: 9 10 24\n"
        "Step 2: 10 - 9 = 1. Remaining: 1 24\n"
        "Step 3: 24 * 1 = 24. Answer: 24",
    ),
    (
        "Problem: 4 9 10 13",
        "Step 1: 13 - 10 = 3. Remaining: 3 4 9\n"
        "Step 2: 9 - 3 = 6. Remaining: 4 6\n"
        "Step 3: 4 * 6 = 24. Answer: 24",
    ),
]


def fewshot_chat_prompt_24(tokenizer, problem: str) -> Tuple[str, bool]:
    """Chat-template prompt with 3-shot Game-24 exemplars, ending with
    'Step 1:' appended to the assistant turn so the model completes in the
    same format as SFT.
    """
    nums = problem.replace(",", " ")
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM_24}]
    for uq, aa in FEWSHOT_EXAMPLES_24:
        msgs.append({"role": "user", "content": uq})
        msgs.append({"role": "assistant", "content": aa})
    msgs.append({"role": "user", "content": f"Problem: {nums}"})
    chat = _apply_chat_template_no_think(tokenizer, msgs)
    # Prime the assistant turn with "Step 1:" so the continuation matches
    # our parser's expectation.
    chat = chat + "Step 1:"
    return chat, False


FEWSHOT_SYSTEM_24_PLAN = (
    "You are a careful arithmetic solver. Use the four given numbers and "
    "basic arithmetic operations (+, -, *, /) to obtain 24. Each number "
    "must be used exactly once. Before each step, emit a planning tag "
    "(one of <PLAN:+>, <PLAN:->, <PLAN:*>, <PLAN:/>, <PLAN:ANS>) that "
    "indicates the operator used next. Respond in the format shown in the "
    "examples, ending with '<PLAN:ANS> Answer: 24'. Do not add any other text."
)

FEWSHOT_EXAMPLES_24_PLAN = [
    (
        "Problem: 4 4 6 8",
        "<PLAN:+> Step 1: 4 + 8 = 12. Remaining: 4 6 12\n"
        "<PLAN:-> Step 2: 6 - 4 = 2. Remaining: 2 12\n"
        "<PLAN:*> Step 3: 2 * 12 = 24. <PLAN:ANS> Answer: 24",
    ),
    (
        "Problem: 2 9 10 12",
        "<PLAN:*> Step 1: 12 * 2 = 24. Remaining: 9 10 24\n"
        "<PLAN:-> Step 2: 10 - 9 = 1. Remaining: 1 24\n"
        "<PLAN:*> Step 3: 24 * 1 = 24. <PLAN:ANS> Answer: 24",
    ),
    (
        "Problem: 4 9 10 13",
        "<PLAN:-> Step 1: 13 - 10 = 3. Remaining: 3 4 9\n"
        "<PLAN:-> Step 2: 9 - 3 = 6. Remaining: 4 6\n"
        "<PLAN:*> Step 3: 4 * 6 = 24. <PLAN:ANS> Answer: 24",
    ),
]


def fewshot_chat_prompt_24_plan(tokenizer, problem: str) -> Tuple[str, bool]:
    """Like fewshot_chat_prompt_24 but with <PLAN:op> tags before each step.

    Does NOT prime 'Step 1:' — the assistant turn starts empty so the model
    can generate the planning token first.
    """
    nums = problem.replace(",", " ")
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM_24_PLAN}]
    for uq, aa in FEWSHOT_EXAMPLES_24_PLAN:
        msgs.append({"role": "user", "content": uq})
        msgs.append({"role": "assistant", "content": aa})
    msgs.append({"role": "user", "content": f"Problem: {nums}"})
    chat = _apply_chat_template_no_think(tokenizer, msgs)
    # No priming — model generates <PLAN:...> then Step 1: etc.
    return chat, False


PROMPT_BUILDERS_24: dict[str, Callable[[object, str], Tuple[str, bool]]] = {
    "sft": sft_prompt_24,
    "fewshot": fewshot_chat_prompt_24,
    "fewshot_plan": fewshot_chat_prompt_24_plan,
}


def get_builder_24(name: str):
    if name not in PROMPT_BUILDERS_24:
        raise ValueError(
            f"Unknown prompt_style '{name}'. Valid: "
            f"{list(PROMPT_BUILDERS_24.keys())}")
    return PROMPT_BUILDERS_24[name]


# ============================================================================
# Countdown prompt builders (N=6, variable target)
# ============================================================================

INSTRUCTION_CD = (
    "Use the six given numbers and integer arithmetic (+, -, *, /) to reach "
    "the target. Each number must be used exactly once. Subtraction must be "
    "non-negative. Division must be exact (no remainder)."
)


def sft_prompt_cd(tokenizer, pool: list[int], target: int) -> Tuple[str, bool]:
    """Raw completion prompt matching our Countdown SFT training distribution."""
    pool_str = " ".join(str(n) for n in sorted(pool))
    text = f"{INSTRUCTION_CD}\n\nProblem: {pool_str} | Target: {target}\nStep 1:"
    return text, True


FEWSHOT_SYSTEM_CD = (
    "You are a careful arithmetic solver for Countdown. Use the six given "
    "numbers and integer arithmetic (+, -, *, /) to reach the target. Each "
    "number must be used exactly once. Subtraction must be non-negative. "
    "Division must be exact (no remainder). Respond with exactly five lines "
    "in the format shown in the examples, ending with 'Answer: {target}' "
    "on the last step. Do not add any other text."
)

FEWSHOT_EXAMPLES_CD = [
    (
        "Problem: 1 6 8 10 10 100 | Target: 252",
        "Step 1: 1 * 6 = 6. Remaining: 6 8 10 10 100\n"
        "Step 2: 10 * 100 = 1000. Remaining: 6 8 10 1000\n"
        "Step 3: 8 + 1000 = 1008. Remaining: 6 10 1008\n"
        "Step 4: 10 - 6 = 4. Remaining: 4 1008\n"
        "Step 5: 1008 / 4 = 252. Answer: 252",
    ),
    (
        "Problem: 1 2 2 3 9 100 | Target: 355",
        "Step 1: 2 * 3 = 6. Remaining: 1 2 6 9 100\n"
        "Step 2: 100 / 2 = 50. Remaining: 1 6 9 50\n"
        "Step 3: 9 + 50 = 59. Remaining: 1 6 59\n"
        "Step 4: 6 * 59 = 354. Remaining: 1 354\n"
        "Step 5: 1 + 354 = 355. Answer: 355",
    ),
    (
        "Problem: 2 2 8 9 10 25 | Target: 627",
        "Step 1: 2 * 8 = 16. Remaining: 2 9 10 16 25\n"
        "Step 2: 9 + 10 = 19. Remaining: 2 16 19 25\n"
        "Step 3: 16 / 2 = 8. Remaining: 8 19 25\n"
        "Step 4: 8 + 25 = 33. Remaining: 19 33\n"
        "Step 5: 19 * 33 = 627. Answer: 627",
    ),
]


def fewshot_chat_prompt_cd(tokenizer, pool: list[int], target: int) -> Tuple[str, bool]:
    """Chat-template prompt with 3-shot Countdown exemplars, ending 'Step 1:'."""
    pool_str = " ".join(str(n) for n in sorted(pool))
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM_CD}]
    for uq, aa in FEWSHOT_EXAMPLES_CD:
        msgs.append({"role": "user", "content": uq})
        msgs.append({"role": "assistant", "content": aa})
    msgs.append({"role": "user",
                 "content": f"Problem: {pool_str} | Target: {target}"})
    chat = _apply_chat_template_no_think(tokenizer, msgs)
    chat = chat + "Step 1:"
    return chat, False


PROMPT_BUILDERS_CD: dict[str, Callable[[object, list[int], int], Tuple[str, bool]]] = {
    "sft": sft_prompt_cd,
    "fewshot": fewshot_chat_prompt_cd,
}


def get_builder_cd(name: str):
    if name not in PROMPT_BUILDERS_CD:
        raise ValueError(
            f"Unknown prompt_style '{name}'. Valid: "
            f"{list(PROMPT_BUILDERS_CD.keys())}")
    return PROMPT_BUILDERS_CD[name]


# ============================================================================
# Generic (task-agnostic, varied-target) prompt builders.
# Pool size and target both vary; examples cover 1/2/3-step trajectories so the
# model can learn to pick its own depth. Format matches our `Step N:` parser.
# ============================================================================

INSTRUCTION_GENERIC = (
    "Combine the given numbers using +, -, *, / to reach the target. "
    "Use each number exactly once."
)

FEWSHOT_SYSTEM_GENERIC = (
    "You are a careful arithmetic solver. Combine the given numbers using "
    "+, -, *, / to reach the target. Each number must be used exactly once. "
    "Write each operation on its own line as 'Step N: a op b = r. "
    "Remaining: ...', and finish with 'Answer: <target>'. If the pool already "
    "equals the target, write only 'Answer: <target>'."
)

FEWSHOT_EXAMPLES_GENERIC = [
    (
        "Numbers: 4 4 6 8 | Target: 24",
        "Step 1: 4 + 8 = 12. Remaining: 4 6 12\n"
        "Step 2: 6 - 4 = 2. Remaining: 2 12\n"
        "Step 3: 2 * 12 = 24. Answer: 24",
    ),
    (
        "Numbers: 3 5 6 | Target: 33",
        "Step 1: 5 * 6 = 30. Remaining: 3 30\n"
        "Step 2: 3 + 30 = 33. Answer: 33",
    ),
    (
        "Numbers: 7 9 | Target: 63",
        "Step 1: 7 * 9 = 63. Answer: 63",
    ),
]


def _format_pool_target(pool, target) -> str:
    nums = " ".join(str(int(n)) for n in sorted(pool))
    return f"Numbers: {nums} | Target: {int(target)}"


def sft_prompt_generic(tokenizer, pool, target) -> Tuple[str, bool]:
    """Raw completion prompt for the varied-target generic task.

    Does not prime 'Step 1:' — a 1-element pool matching the target means
    the correct continuation is 'Answer: <target>', not a wasted Step 1.
    """
    head = _format_pool_target(pool, target)
    text = f"{INSTRUCTION_GENERIC}\n\n{head}\n"
    return text, True


def fewshot_chat_prompt_generic(tokenizer, pool, target) -> Tuple[str, bool]:
    """Chat-template prompt with varied-length exemplars. No 'Step 1:' prime
    so trivial pools (length-1) can emit 'Answer:' directly."""
    head = _format_pool_target(pool, target)
    msgs = [{"role": "system", "content": FEWSHOT_SYSTEM_GENERIC}]
    for uq, aa in FEWSHOT_EXAMPLES_GENERIC:
        msgs.append({"role": "user", "content": uq})
        msgs.append({"role": "assistant", "content": aa})
    msgs.append({"role": "user", "content": head})
    chat = _apply_chat_template_no_think(tokenizer, msgs)
    return chat, False


PROMPT_BUILDERS_GENERIC: dict[str, Callable[[object, list, int], Tuple[str, bool]]] = {
    "sft": sft_prompt_generic,
    "fewshot": fewshot_chat_prompt_generic,
}


def get_builder_generic(name: str):
    if name not in PROMPT_BUILDERS_GENERIC:
        raise ValueError(
            f"Unknown prompt_style '{name}'. Valid: "
            f"{list(PROMPT_BUILDERS_GENERIC.keys())}")
    return PROMPT_BUILDERS_GENERIC[name]
