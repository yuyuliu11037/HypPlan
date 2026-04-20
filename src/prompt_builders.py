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
    chat = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    # Prime the assistant turn with "Step 1:" so the continuation matches
    # our parser's expectation.
    chat = chat + "Step 1:"
    return chat, False


PROMPT_BUILDERS_24: dict[str, Callable[[object, str], Tuple[str, bool]]] = {
    "sft": sft_prompt_24,
    "fewshot": fewshot_chat_prompt_24,
}


def get_builder_24(name: str):
    if name not in PROMPT_BUILDERS_24:
        raise ValueError(
            f"Unknown prompt_style '{name}'. Valid: "
            f"{list(PROMPT_BUILDERS_24.keys())}")
    return PROMPT_BUILDERS_24[name]
