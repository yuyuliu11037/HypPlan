# Agent brief — 4-model × 3-method × 8-dataset baseline sweep

You are running a baseline experiment grid and writing a final report.
This document is **self-contained**: everything you need is in here.
Do not assume a particular existing codebase. You will receive the 8
test JSONL files separately; their schemas are specified in §4.

---

## 0. Environment setup

You build the pipeline from scratch in a fresh Python project. You
need:

- **Python**: 3.10 or 3.11 (3.12 may hit `bitsandbytes` issues).
- **CUDA**: 12.1 or 12.4 toolkit + matching NVIDIA driver.
- **GPU**: see §5 for memory sizing.
- **HuggingFace account + access token** for `meta-llama/Llama-3.3-70B-Instruct`
  (gated). Accept the license on the model page first, then set
  `export HF_TOKEN=hf_xxx` in your shell. The other 3 models are
  open-access.

Recommended `pip install` (versions known to work as of 2026 Q2 — bump
forward if needed, but pin in the report):

```bash
pip install --upgrade pip
pip install \
    "torch>=2.4" \
    "transformers>=4.45" \
    "accelerate>=0.34" \
    "bitsandbytes>=0.43" \
    "sentencepiece" \
    "tokenizers>=0.20" \
    "tqdm" \
    "numpy"
```

Project layout (suggestion — feel free to reorganize):

```
.
├── data/                       # the 8 test JSONL files (provided by user)
├── src/
│   ├── load_model.py           # §2 model loader
│   ├── prompts.py              # §4 per-task prompt builders
│   ├── score.py                # §4 per-task scorers + answer-key extractors
│   ├── tot_adapters.py         # §4 per-task ToT adapters
│   ├── run_fewshot.py          # §3.1 entry point
│   ├── run_sc.py               # §3.2 entry point
│   ├── run_tot.py              # §3.3 entry point
│   └── update_summary.py       # §6.2/6.3 summary regenerator
├── results/multimodel_v2/      # outputs (§6)
└── logs/multimodel_v2/         # stdout/stderr per cell
```

A reference end-to-end code skeleton is in **Appendix A** at the
bottom of this document.

---

## 1. Goal

For every cell in this 4 × 3 × 8 grid, run the test set and record
accuracy, generated-token count, and wall-clock latency.

| Axis    | Values                                                                                |
|---------|---------------------------------------------------------------------------------------|
| Model   | Phi-4 14B • Qwen3-14B (non-thinking) • Gemma-3 27B-IT • Llama-3.3 70B-Instruct        |
| Method  | Few-shot • Tree-of-Thoughts (ToT) • Self-Consistency (SC)                             |
| Dataset | g24 • prontoqa (pq) • blocksworld (bw) • graphcolor (gc) • numpath • rulechain • clutrr • proofwriter |

= 96 cells. Skip a cell only if its result file is already complete
(see §6.4).

---

## 2. Models

Use the latest official Hugging Face IDs. Do not substitute older
versions, even if a newer one is hard to load — escalate instead.

| Tag         | HF ID                                          | Approx. memory (4-bit) | Loading notes                                                     |
|-------------|------------------------------------------------|------------------------|-------------------------------------------------------------------|
| `phi4_14b`  | `microsoft/phi-4`                              | ~10 GB                 | Standard chat template.                                            |
| `qwen3_14b` | `Qwen/Qwen3-14B`                               | ~10 GB                 | **Non-thinking mode.** Pass `enable_thinking=False` to `tokenizer.apply_chat_template`. Verify outputs contain no `<think>` tokens before launching the full sweep. |
| `gemma3_27b`| `google/gemma-3-27b-it`                        | ~17 GB                 | Chat template does not accept a separate `system` role — fold the system message into the first user turn (see §3.0). |
| `llama3_70b`| `meta-llama/Llama-3.3-70B-Instruct`            | ~35 GB                 | Use `device_map="auto"`. Spans 1 or more GPUs depending on your per-device memory; size the allocation so weights + activations + KV cache all fit. |

All four are used **frozen** — no LoRA, no fine-tuning.

### 2.1 Reference loader (use this exact pattern)

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)

def load_model(hf_id: str):
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # required for batched decode

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tok
```

`device_map="auto"` spreads weights across all visible GPUs as needed.
Set `CUDA_VISIBLE_DEVICES` per-process to pin which GPUs each cell
uses.

### 2.2 Building chat-template prompts

For all models **except Gemma-3-27B-IT**, build prompts as a list of
chat messages and apply the tokenizer's chat template:

```python
messages = [
    {"role": "system", "content": SYSTEM_TEXT},
    {"role": "user",   "content": USER_TEXT},
]
prompt = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,    # only honored by Qwen3 — ignored by others
)
input_ids = tok(prompt, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(model.device)
# add_special_tokens=False matters: chat-template already adds BOS.
```

For **Gemma-3-27B-IT**, fold the system message into the first user
turn (Gemma's chat template does not accept a `system` role):

```python
if "gemma-3" in hf_id.lower():
    user_with_sys = SYSTEM_TEXT + "\n\n" + USER_TEXT
    messages = [{"role": "user", "content": user_with_sys}]
else:
    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user",   "content": USER_TEXT},
    ]
```

For tasks where §4 only specifies a `user` prompt (no system), pass
just the user message — same pattern, both branches collapse.

Confirm `transformers >= 4.45` (Gemma-3 / Phi-4 / Llama-3.3 support).
Pin the exact versions used in the final report.

---

## 3. Methods

All three methods consume the **same per-task prompt** (defined in §4).
They differ only in how they decode and aggregate.

### 3.0 Shared decoding rules (apply to all methods)

- **Greedy decode** = `do_sample=False`. Sampling = `do_sample=True,
  temperature=0.7, top_p=0.95`.
- Always pass `pad_token_id=tok.eos_token_id` to `model.generate(...)`.
- Always wrap propose/value/full prompts through the §2.2 chat-template
  builder. ToT's propose and value calls are **also** chat-wrapped —
  they are LLM calls just like the main decode.
- Decode the model's emission with `tok.decode(out_ids[input_len:],
  skip_special_tokens=True)`.
- For **SC** (K samples), do **one** `model.generate(...)` call with
  `do_sample=True, num_return_sequences=K`. This is faster than K
  separate calls and uses the same input KV cache once.
- For **ToT**, batch the propose calls across active states at each
  depth, and the value calls across candidates. A batch size of 4–8
  works well. Use `tokenizer(..., padding=True, truncation=True,
  max_length=2048)` for the batched inputs (left padding, see §2.1).
- Reproducibility: at process start, run `torch.manual_seed(1234)`.
  This makes the SC and ToT sampling deterministic per-process.
- `max_new_tokens`: 384 for the main decode (Few-shot, SC, ToT
  terminal-trajectory scoring); 80 for ToT propose; 20 for ToT value.

### 3.1 Few-shot (greedy)

Single greedy decode with `max_new_tokens=384`.

```
prompt  = build_prompt(task, record)
output  = greedy_decode(model, prompt, max_new_tokens=384)
correct = score(task, output, record)   # see §4 per-task scorers
```

Record `top1_ok = correct`.

> The name "few-shot" is used because for some tasks the prompt
> already includes 1–3 in-context worked examples (see §4). For tasks
> where the prompt has no examples, "few-shot" here just means "the
> standard instruction prompt with greedy decoding" — i.e. zero-shot
> CoT. Keep the name `fewshot` for consistency in result filenames.

### 3.2 Self-Consistency (SC)

Wang et al. 2023. Sample K=5 trajectories at T=0.7, top-p=0.95, then
take the **majority vote on a task-specific answer key** (defined in
§4 per task).

```
prompt   = build_prompt(task, record)
samples  = sample_decode(model, prompt, K=5, T=0.7, top_p=0.95,
                         max_new_tokens=384)
keys     = [extract_answer_key(task, s) for s in samples]
top_key, _ = Counter(k for k in keys if k is not None).most_common(1)[0]
# A sample "wins" if it has the top key AND it scores correct.
majority_ok = any(score(task, s, record) and keys[i] == top_key
                  for i, s in enumerate(samples))
```

Record `majority_ok` and (for diagnostics) `any_ok = any(score(...))`.

### 3.3 Tree-of-Thoughts (ToT)

Paper-faithful BFS from Yao et al. 2023. Same model is used for both
the **proposer** and the **value model**.

Hyperparameters (locked):
- `n_generate = 1` proposer call per active state per depth
- proposer is asked to output **3 candidate next steps** in a single
  call (see §4 per-task propose prompts)
- `n_evaluate = 3` value-model calls per candidate state
- `n_select = 5` top states kept per depth (BFS beam width)
- `temperature = 0.7` for both propose and value
- `propose_max_new_tokens = 80`, `value_max_new_tokens = 20`
- `max_depth` is task-specific (see §4)

Per-step value scoring follows the ToT paper:
- value-model output containing the word `sure` → score 20
- containing `likely` → score 1
- containing `impossible` → score 0.001
- otherwise → 0

For each state, sum the value scores from its `n_evaluate` calls.
Keep the top `n_select` states by total value at every depth. After
`max_depth` (or once all active states are terminal), report the
`top1_ok` accuracy of the **highest-value remaining state** under the
task scorer.

Pseudocode:

```python
def tot_solve(adapter, model, tok):
    active = [adapter.init_partial()]
    for depth in range(adapter.max_depth):
        terminal = [y for y in active if adapter.is_terminal(y)]
        live     = [y for y in active if not adapter.is_terminal(y)]
        if not live:
            active = terminal
            break

        # Propose
        candidates = []
        for y in live:
            gen = sample(model, adapter.propose_prompt(y), T=0.7,
                         max_new=80)
            for step in adapter.extract_steps(gen, y):
                candidates.append(y + ("\n" if y else "") + step)
        candidates = dedup(candidates) + terminal

        # Value
        scored = []
        for c in candidates:
            v = sum(value_score(sample(model, adapter.value_prompt(c),
                                       T=0.7, max_new=20))
                    for _ in range(3))
            scored.append((c, v))

        active = [c for c, _ in sorted(scored, key=lambda p: -p[1])][:5]

    best = active[0] if active else ""
    return adapter.is_correct(best)
```

You build a **per-task ToT adapter** that exposes:
- `init_partial()` — initial empty trajectory string
- `propose_prompt(partial)` — prompt string asking for 3 candidate next
  steps
- `extract_steps(gen, partial)` — parse 0+ candidate next-step strings
  from the proposer's output
- `value_prompt(partial)` — prompt asking for `sure / likely /
  impossible`
- `is_terminal(partial)` — does the partial trajectory have an answer?
- `is_correct(partial)` — does it score correct under the §4 scorer?
- `max_depth` — depth budget

Adapter recipes per task are in §4.

---

## 4. The 8 datasets

The "8 dataset JSONL files" are eight test sets — one reasoning task
each — that the user (Yuyu) will share with you as a separate
download. They are **JSON-Lines** files: one JSON object per line,
each object is a single test record. Read them with:

```python
records = [json.loads(line) for line in open(path)]
```

You do **not** need to construct or download these datasets yourself
— they are already curated and frozen. Sources, in case it helps:
g24 (Game-of-24, an arithmetic puzzle), pq (ProntoQA, synthetic
deductive reasoning), bw (Blocksworld from PlanBench, symbolic
planning), gc (Graph 3-Coloring, in-house generated), and four
in-house generated logical/arithmetic tasks: numpath (number-path
reachability), rulechain (Horn-clause forward chaining), clutrr
(kinship composition, CLUTRR-style), proofwriter (closed-world
deductive reasoning, ProofWriter depth-3).

Each record has at least:
- A field describing the problem in structured form (e.g., a graph,
  a list of facts, a starting number).
- A `prompt` field — pre-built natural-language statement of the
  problem. For some tasks the `prompt` already contains 1–3 in-context
  worked examples (PQ / BW / GC); for others it is just the task
  statement (numpath, rulechain, clutrr, proofwriter, g24). §4.1–4.8
  spells out which.
- A gold answer field used **only by the scorer**, never given to the
  model: `answer_label`, or task-specific fields such as `answer`,
  `gold_solution`, `target`.

Apply the `Limit` column (head N rows) **before** running the model
— that is the row count the user wants reported. Do not shuffle.

| Task tag       | Reasoning type                                    | Test file                  | Records in file | Limit (use first N) | ToT max_depth |
|----------------|---------------------------------------------------|----------------------------|-----------------|---------------------|---------------|
| `g24`          | Arithmetic puzzle (combine 4 numbers → 24)        | `24_test.jsonl`            | 956             | 100                 | 3             |
| `pq`           | Deductive reasoning (true/false from rules)       | `prontoqa_test.jsonl`      | 200             | 100                 | 8             |
| `bw`           | Symbolic planning (block stacking)                | `blocksworld_test.jsonl`   | 200             | 100                 | 16            |
| `gc`           | CSP (3-color a small graph)                       | `graphcolor_test.jsonl`    | 200             | 100                 | 6 (= n)       |
| `numpath`      | Arithmetic search (reach target via fixed ops)    | `numpath_test.jsonl`       | 200             | 200                 | 6             |
| `rulechain`    | Forward chaining (derive a target predicate)      | `rulechain_test.jsonl`     | 600             | 200                 | 8             |
| `clutrr`       | Kinship-relation composition                      | `clutrr_test.jsonl`        | 200             | 200                 | 4             |
| `proofwriter`  | Closed-world theorem proving (true/false)         | `proofwriter_test.jsonl`   | 200             | 200                 | 8             |

**Do not regenerate, re-shuffle, or re-split the test data.** If a
file is missing or has fewer rows than the "Records in file" column,
stop and ask the user.

Below is the exact schema of each file, the prompt-build rule for
Few-shot/SC, the scoring rule, and the ToT adapter recipe.

### 4.1 Game-of-24 (`g24`)

**Schema** (each row):
```jsonc
{
  "problem": "1,4,4,12",        // four numbers, comma-separated
  "text":    "Problem: 1 4 4 12\nStep 1: ...",  // gold trajectory
  "step_offsets": [18, 57, 95]
}
```

**Few-shot / SC prompt** (chat-template):
- system: "You will solve a Game-of-24-like puzzle. Use each number
  exactly once and the four operations (+, -, *, /) to make the
  target. Output one operation per step. Output format: `Step 1: a op
  b = r`, `Step 2: c op d = s`, ..., `Answer: <target>`."
- user: `Numbers: 1 4 4 12\nTarget: 24`

**Scorer.** Parse the model output line by line: `Step N: a op b = r`.
For each step, verify the arithmetic and that `a` and `b` are in the
current pool. Replace `a`,`b` with `r`. Final pool must be exactly
`[24]`. The first step's pool is the input four numbers; each input
number must be used exactly once across steps.

**SC answer key.** Regex `Answer\s*:\s*(-?\d+)` → integer. Default
`None` if no match.

**ToT adapter.**
- `init_partial = ""`
- propose prompt: "Input: <current pool>\nPossible next steps:\n<list
  examples like in the ToT G24 paper, one per line, of form `a op b =
  r (left: ...)`>". Output 3 candidates.
- `extract_steps`: regex each line for `(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)\s*\(left:\s*([^)]*)\)`. Validate arithmetic + pool membership.
- `value_prompt`: "Evaluate if given numbers can reach 24
  (sure/likely/impossible)\n<remaining pool>". After `max_depth=3`
  steps, terminal if pool is `[24]`.
- `is_correct`: full G24 scorer above on the assembled trajectory.

### 4.2 ProntoQA (`pq`)

**Schema:**
```jsonc
{
  "id": "ProntoQA_32",
  "prompt": "<full 3-shot prompt ending with 'Answer:'>",
  "answer_label": "A" | "B",            // gold
  "init_state_text": "Initial fact: ... Question: is X Y?"
}
```

The `prompt` field already contains 3 in-context examples followed by
the test query and a trailing `Answer:`. Use it verbatim.

**Few-shot / SC prompt.** Wrap `prompt` as the user turn (no system
message — Phi-4/Llama can take a one-line system like "Reply with
exactly one letter: A or B.", but for parity put it in the user turn
as well for Gemma).

**Scorer.** Look in the **first non-empty line** of the generation
for `\b(A|B)\b`. Match against `answer_label`. Anything else = wrong.

**SC answer key.** Same regex; the letter itself is the key.

**ToT adapter.**
- `init_partial = ""`
- `max_depth = 8`
- propose: "Propose 3 next derivation steps for the partial
  derivation below. Each line must be either `Step N: <derived fact>.`
  or `Answer: A`/`Answer: B`. <rules + facts + partial>"
- `value_prompt`: "Will continuing the partial derivation reach the
  correct answer? Reply with one of: sure/likely/impossible."
- `is_terminal`: trajectory contains an `Answer: ` line.
- `is_correct`: extracted answer letter equals `answer_label`.

### 4.3 Blocksworld (`bw`)

**Schema:**
```jsonc
{
  "id": 334,
  "prompt": "<full 1-shot PDDL-style prompt ending with '[PLAN]\n'>",
  "answer_label": "(unstack blue orange)\n(stack blue yellow)\n..."
}
```

The `prompt` already contains a worked example and the test
[STATEMENT] block, ending with `[PLAN]\n`. Use it verbatim.

**Few-shot / SC prompt.** Wrap as user turn.

**Scorer (goal-reaching, not exact-match).** Parse the generation
into action lines `(<action> <args>)` until the first non-action line
or `[PLAN END]`. Replay actions on the initial state encoded in the
prompt's `As initial conditions ...` text. An action is one of
`(pick-up X)`, `(put-down X)`, `(stack X Y)`, `(unstack X Y)` with
standard blocksworld preconditions. Score = 1 iff after all actions
the goal facts (parsed from `My goal is to have that ...`) are all
satisfied. Illegal actions abort the simulation as a failure.

**SC answer key.** The tuple of canonical action strings (e.g.
`("(unstack blue orange)", "(stack blue yellow)", ...)`).

**ToT adapter.**
- `init_partial = ""`
- `max_depth = 16` (blocksworld plans can be long).
- propose: "Propose 3 next single-action lines for this partial plan.
  One line per candidate, format `(<action> <args>)`."
- `value_prompt`: "Will this partial plan reach the goal?
  sure/likely/impossible." Treat illegal-action partials as
  `impossible` automatically (skip the LLM call).
- `is_terminal`: simulator reaches goal OR last action illegal.
- `is_correct`: simulator goal reached.

### 4.4 Graph Coloring (`gc`)

**Schema:**
```jsonc
{
  "id": "gc_0",
  "n": 6,
  "edges": [[0,1], [0,4], ...],
  "prompt": "<full 2-shot prompt ending with 'Coloring:'>",
  "init_state_text": "Graph 3-coloring task. Vertices: ... Edges: ...",
  "gold_solution": ["R","G","B","R","B","G"]      // one valid coloring
}
```

**Few-shot / SC prompt.** Use `prompt` verbatim as the user turn.

**Scorer.** Parse generation lines `V<i> = <color>` (color in
{red/green/blue or R/G/B}) into a dict `i -> color`. Must cover all
`n` vertices. Score = 1 iff every edge `(u,v)` has different colors.
Note: `gold_solution` is just one valid coloring — any valid coloring
counts.

**SC answer key.** The tuple of (vertex_id, color) pairs sorted by
vertex_id, or `None` if not all vertices are covered.

**ToT adapter.**
- `init_partial = ""`
- `max_depth = n` (one vertex colored per step)
- propose: "Propose 3 candidate color assignments for the next
  uncolored vertex (the smallest index not yet colored). Output
  exactly 3 lines of form `V<i> = <color>` for the same vertex."
- `value_prompt`: "Is the partial coloring valid (no edge conflict
  yet) AND extendable to a complete valid coloring?
  sure/likely/impossible."
- `is_terminal`: all `n` vertices colored, or first conflict.
- `is_correct`: full graphcolor scorer on the partial.

### 4.5 Number-path (`numpath`)

**Schema:**
```jsonc
{
  "id": "numpath_test_0",
  "start": 17,
  "target": 36,
  "ops": [{"kind":"SUB","const":2}, {"kind":"DIV","const":3}, ...],
  "max_value": 999,
  "n_steps": 3,
  "prompt": "Apply operations to transform the start number into the target number. Use any sequence of the allowed operations.\nStart: 17    Target: 36\nOperations: - 2, / 3, + 3, * 2",
  "answer_label": "Step 1: 17 - 2 = 15\nStep 2: 15 + 3 = 18\nStep 3: 18 * 2 = 36\nAnswer: 36"
}
```

There are no in-context examples in `prompt`. The `ops` list is the
only set of operations the model may use; it may apply each op any
number of times in any order, but only those four constants. `DIV` is
integer division and only legal when divisor evenly divides current
value.

**Few-shot / SC prompt.**
- system: "Solve the number-path puzzle. Output one step per line in
  the form `Step N: <current> <op> <const> = <next>`, then `Answer:
  <final>`. You may use any sequence of the allowed operations; each
  may be used any number of times. Stop after `Answer:`."
- user: contents of the `prompt` field.

**Scorer.** Parse `Step N: a op b = r` lines. For each step, `op b`
must be one of the allowed ops (match `kind` and `const`). `a` must
equal the current value. Verify arithmetic. Final answer parsed from
`Answer: <int>` must equal `target` AND must equal the last step's
`r`. Intermediate values must stay in `[0, max_value]`.

**SC answer key.** The integer after `Answer:` (or final step's `r`
if no `Answer:` line).

**ToT adapter.**
- `init_partial = ""`
- `max_depth = 6`
- propose: "Propose 3 candidate next single-step lines `<current> <op>
  <const> = <next>` using the allowed operations. Stay in [0, 999]."
- `value_prompt`: heuristic — compute `dist = |current − target|`. If
  dist=0 emit `sure`; if `dist <= |target|/2` emit `likely`; else
  `impossible`. (You may skip the LLM call for value if you compute
  this directly — record the choice.)
- `is_terminal`: current value equals target, or step count = 6.
- `is_correct`: scorer above on the assembled trajectory.

### 4.6 Rule-chaining (`rulechain`)

**Schema:**
```jsonc
{
  "id": "rulechain_test_0",
  "initial_facts": ["p10","p2","p3","p6"],
  "target": "p14",
  "rules": [{"premises":["p10","p14"], "conclusion":"p5"}, ...],
  "n_steps": 2,
  "prompt": "Rules:\n- if p10 and p14, then p5\n- ...\nInitial facts: p10, p2, p3, p6\nGoal: derive p14",
  "answer_label": "Step 1: apply rule: if p2, then p9\nStep 2: apply rule: if p9, then p14\nAnswer: p14 is derived."
}
```

**Few-shot / SC prompt.**
- system: "Forward-chain over the rules to derive the goal predicate.
  Output one rule application per step in the form `Step N: apply
  rule: if <premises>, then <conclusion>`, then `Answer: <goal> is
  derived.` Use only the rules given. Stop at the answer line."
- user: contents of `prompt`.

**Scorer.** Parse `Step N: apply rule: if <premise list>, then
<conclusion>` lines. For each step, match the (premises, conclusion)
to one of the `rules` entries. Track the running set of derived facts
(starting from `initial_facts`). Each step's premises must be a
subset of currently-derived facts; the step's conclusion is added to
that set. Final answer line `Answer: <pred> is derived.` must match
`target`, and `target` must be in the final derived set.

**SC answer key.** The predicate after `Answer:` (regex
`Answer\s*:\s*(\S+)`).

**ToT adapter.**
- `init_partial = ""`
- `max_depth = 8`
- propose: "Propose 3 candidate next forward-chaining rule
  applications. Each line: `apply rule: if <prem>, then <conc>` using
  rules and the current derived set."
- `value_prompt`: "Can the goal predicate `<target>` be derived by
  continuing this partial chain? sure/likely/impossible."
- `is_terminal`: target ∈ derived set.
- `is_correct`: same.

### 4.7 CLUTRR-like kinship (`clutrr`)

**Schema:**
```jsonc
{
  "id": "clutrr_test_0",
  "k": 4,
  "entities": ["Ivy","Bob","Ben","Zach","Xavier","Fred","Olivia"],
  "edges": [[0,"father",1],[2,"brother",3], ...],
  "query": [0, 4],                    // (subject_idx, object_idx)
  "answer": "first-cousin-once-removed",
  "chain": ["father","father","brother","son"],
  "prompt": "Ivy is the father of Bob.\n...\n\nHow is Ivy related to Xavier?",
  "answer_label": "Step 1: ...\nStep 4: Ivy is the first-cousin-once-removed of Xavier\nAnswer: Ivy is the first-cousin-once-removed of Xavier."
}
```

**Few-shot / SC prompt.**
- system: "You are given family relationships. Compose them step by
  step to answer the question. Output one step per line in the form
  `Step N: <subject> is the <relation> of <intermediate>`, then end
  with `Answer: <subject> is the <relation> of <object>.`"
- user: `prompt`.

**Scorer.** Extract the predicted relation token from the final
`Answer:` line (e.g. `first-cousin-once-removed`). Compare to
`answer` field. Accept hyphenated and space variants
(`first cousin once removed` ↔ `first-cousin-once-removed`).

**SC answer key.** The normalized relation token from the `Answer:`
line.

**ToT adapter.**
- `init_partial = ""`
- `max_depth = 4`
- propose: "Propose 3 candidate next composition steps in the form
  `Step N: <subj> is the <relation> of <next entity>`."
- `value_prompt`: "Will this partial composition reach the queried
  relation between <subject> and <object>?
  sure/likely/impossible."
- `is_terminal`: trajectory contains `Answer:`.
- `is_correct`: scorer above.

### 4.8 ProofWriter (`proofwriter`)

**Schema:**
```jsonc
{
  "id": "proofwriter_test_0",
  "theory_text": "The cat is nice. ... If someone visits the cat ...",
  "initial_facts": [["cat","is","nice","+"], ...],
  "rules_struct": {"rule1": {...}, ...},
  "target": ["mouse","is","young","+"],
  "target_text": "The mouse is young.",
  "answer": true,
  "QDep": 0,
  "prompt": "<theory> Question: <target_text> Is the statement above true or false?",
  "answer_label": "Answer: True"     // or with proof steps
}
```

This is **closed-world-assumption depth-3** — answer is `True` iff
`target` is derivable from the theory under CWA, else `False`.

**Few-shot / SC prompt.**
- system: "You are given a theory and a statement. Decide whether the
  statement is True or False under closed-world assumption. You may
  output reasoning steps but the **last line** must be exactly
  `Answer: True` or `Answer: False`."
- user: `prompt`.

**Scorer.** Find the last `Answer:` line, normalize to `True`/`False`
(case-insensitive), compare to the boolean `answer`.

**SC answer key.** The normalized True/False string.

**ToT adapter.**
- `init_partial = ""`
- `max_depth = 8`
- propose: "Propose 3 candidate next inference steps. Each line: one
  fact derived by applying a rule to currently known facts. Or output
  `Answer: True` / `Answer: False` if you can already decide."
- `value_prompt`: "Can the statement be decided True/False from this
  partial derivation? sure/likely/impossible."
- `is_terminal`: trajectory contains an `Answer:` line.
- `is_correct`: extracted answer matches `answer`.

---

## 5. Hardware notes

This brief makes no assumption about your hardware. Use whatever GPUs
you have. The numbers below are guidance for sizing, not requirements.

- **Memory sizing (4-bit NF4)**: Phi-4 14B and Qwen3-14B ≈ ~10 GB
  each; Gemma-3 27B-IT ≈ ~17 GB; Llama-3.3-70B-Instruct ≈ ~35 GB.
  These are weights only — add ~2–4 GB headroom per active forward
  pass for activations + KV cache at the 384-token decode budget,
  more for ToT (it batches many proposer/value calls).
- **Single- vs multi-GPU**: any model that fits on one GPU should run
  on one GPU. For Llama-3.3-70B-Instruct, if it does not fit on a
  single device at 4-bit, shard it with
  `device_map="auto"` across the smallest number of devices that
  give it enough free memory.
- **Parallelism**: each `(model, task, mode)` cell is an independent
  process. Run as many cells in parallel as your hardware allows.
  Independent processes write to distinct output files (see §6.1) so
  there is no contention.
- **Sharding inside a cell**: for the slow cells (any model × ToT,
  plus 70B × anything), shard the test set across N processes by
  record index (`record_idx % N == rank`), write per-shard JSONLs,
  and concatenate at the end. ToT with the 70B model will be the
  longest cell — plan accordingly.
- **Software pin**: record the exact `transformers`, `torch`,
  `bitsandbytes`, CUDA, and GPU model in the final report's
  reproducibility block.

If your hardware cannot fit Llama-3.3-70B-Instruct in 4-bit at all,
stop and escalate before continuing — do not silently swap in a
smaller Llama variant.

---

## 6. Result organization

Place all outputs under `results/multimodel_v2/`. Logs under
`logs/multimodel_v2/`.

### 6.1 Per-cell JSONL files

```
results/multimodel_v2/{tag}_{task}_{mode}.jsonl    one line per record
logs/multimodel_v2/{tag}_{task}_{mode}.log         stdout/stderr
```

`{tag}` ∈ {`phi4_14b`, `qwen3_14b`, `gemma3_27b`, `llama3_70b`}.
`{task}` ∈ the 8 tags above.
`{mode}` ∈ {`fewshot`, `tot`, `sc`}.

Per-record JSONL row:

```jsonc
{
  "id": "<rec id or row index>",
  "task": "pq",
  "mode": "fewshot" | "tot" | "sc",
  "top1_ok": true | false,                      // fewshot/tot: greedy correctness
  "majority_ok": true | false | null,           // sc only
  "any_ok": true | false | null,                // sc/tot diagnostic: any-of-K correct
  "top1_gen": "Step 1: ...",                    // first 600 chars of greedy/tot output
  "sample_gens": ["...", "..."],                // sc/tot: K samples (first 600 chars each)
  "n_gen_tokens": 173,                          // total tokens generated for this record
  "latency_s": 4.21                             // wall-clock for this record
}
```

### 6.2 Summary file

After every cell finishes, append/update one row in
`results/multimodel_v2/summary.jsonl`:

```json
{"tag":"phi4_14b","task":"pq","mode":"fewshot",
 "n":100,"correct":62,"acc":0.62,
 "tokens_total":17321,"latency_total_s":820,
 "timestamp":"2026-04-29T14:31:00Z"}
```

For `mode=sc`, `correct` = count of `majority_ok=true`. For
`mode=fewshot` or `mode=tot`, `correct` = count of `top1_ok=true`.

### 6.3 Summary markdown — `results/multimodel_v2/summary.md`

Three tables, one per method, each 4 rows × 8 columns, regenerated
from `summary.jsonl` after every cell:

```
## Few-shot (greedy, 1 decode/problem)

| Model           | g24 | pq  | bw  | gc  | numpath | rulechain | clutrr | proofwriter |
|-----------------|----:|----:|----:|----:|--------:|----------:|-------:|------------:|
| Phi-4 14B       |     |     |     |     |         |           |        |             |
| Qwen3-14B (n/t) |     |     |     |     |         |           |        |             |
| Gemma-3 27B-IT  |     |     |     |     |         |           |        |             |
| Llama-3.3 70B   |     |     |     |     |         |           |        |             |

## ToT BFS (n_generate=1, n_evaluate=3, n_select=5, T=0.7) — top-1 reported

(same shape)

## Self-Consistency (K=5, T=0.7, majority vote)

(same shape)
```

Cells: `correct/n` percentage, e.g. `62/100 = 62%`. Empty cell if not
yet run; `FAIL` if the run errored out.

### 6.4 Skip rule

Before launching a cell, check:

```bash
out="results/multimodel_v2/${tag}_${task}_${mode}.jsonl"
expected_rows=<the limit from §4 table>
if [ -s "$out" ] && [ "$(wc -l < "$out")" -ge "$expected_rows" ]; then
  echo "[skip] $tag $task $mode (already complete)"; continue
fi
```

### 6.5 Final report — `results/multimodel_v2/REPORT.md`

Write/update after every batch (do not wait until all 96 cells
finish). Sections:

1. **Headline tables** — paste the three tables from §6.3.
2. **Per-task winner** — one bullet per task: which (model, method)
   achieves the highest accuracy on that task, and the gap (in
   percentage points) over the strongest few-shot baseline. One
   sentence each.
3. **Compute summary** — total generated tokens and total wall-clock
   latency per model (sum across all 24 cells of that model). Plain
   language, no academic prose.
4. **Anomalies / failed cells** — list every cell that errored,
   OOM'd, produced format-broken output (e.g. Qwen3 emitted `<think>`
   blocks despite the flag), or ran with a configuration deviation.
   Be specific.
5. **Reproducibility block** — exact `transformers`, `torch`, and
   `bitsandbytes` versions; the GPUs each cell ran on (extract from
   logs); seed = 1234 for sampling RNG.

---

## 7. Pre-flight checklist

Before launching the full grid:

1. All 8 test JSONL files are present and have the row counts in §4.
2. All 4 model IDs download and load successfully (single-record
   smoke test for each — feed one record from each task and verify a
   non-empty generation).
3. Qwen3 non-thinking mode is verified: 5 sample generations contain
   no `<think>` tokens.
4. Gemma-3 chat-template handles the merged-system-into-user prompt
   without crashing.
5. Llama-3.3-70B-Instruct loads in 4-bit (on whatever number of GPUs
   you allocate to it) and produces a sane completion on one g24
   record.
6. The 4 ToT adapters that you wrote (numpath, rulechain, clutrr,
   proofwriter) round-trip a gold trajectory on 5/5 sampled records
   (i.e. their `is_correct` returns True for the trajectory in
   `answer_label`).
7. You have enough total GPU memory + compute time to run the sweep.
   If hardware cannot host Llama-3.3-70B in 4-bit at all, escalate
   before starting.

If any item fails, stop and report the failure.

---

## 8. Run order

Cheapest-first:

1. **Few-shot** across all 4 models × 8 tasks (32 cells).
2. **SC** across all 4 models × 8 tasks (32 cells).
3. **ToT** across all 4 models × 8 tasks (32 cells, slowest).

Within each method, run smaller models first (Phi-4 → Qwen3 → Gemma
→ Llama) so configuration problems surface quickly.

Decoding seed: `torch.manual_seed(1234)` per process for
reproducibility of the SC/ToT sampling. Note this in the report.

---

## 9. What "done" means

- All 96 per-cell JSONLs exist with the expected row counts.
- `summary.jsonl`, `summary.md`, and `REPORT.md` are written and
  current.
- A short message back to the user that says: "done; N cells
  finished, M failed, see `results/multimodel_v2/REPORT.md`", with
  one headline finding (e.g. "Llama-3.3 70B + SC tops 6/8 tasks;
  Gemma-3 27B + ToT wins on graph-coloring").

Do not change the methodology, the data, or the scoring without
asking.

---

## Appendix A. Reference end-to-end code skeleton

This sketch shows how the pieces fit. It is illustrative — fill in
the per-task `build_prompt` / `score_one` / `extract_answer_key` /
ToT adapters from §4. Run as one process per `(model, task, mode)`
cell.

```python
# run_cell.py
import argparse, json, time, os, sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------- Model loader (§2.1) ----------

def load_model(hf_id):
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16,
    ).eval()
    return m, tok

# ---------- Chat-template builder (§2.2) ----------

def build_chat_prompt(tok, hf_id, system_text, user_text):
    if "gemma-3" in hf_id.lower():
        msgs = [{"role": "user",
                 "content": (system_text + "\n\n" + user_text)
                            if system_text else user_text}]
    else:
        msgs = []
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        msgs.append({"role": "user", "content": user_text})
    kwargs = {}
    if "qwen3" in hf_id.lower():
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **kwargs)

# ---------- Per-task hooks you must implement from §4 ----------

def task_prompt(task, rec):
    """Return (system_text, user_text) for this record. See §4."""
    ...

def score_one(task, generation, rec) -> bool:
    """Return True if the generation solves the record. See §4."""
    ...

def extract_answer_key(task, generation, rec):
    """Return a hashable key for SC majority vote, or None. See §4."""
    ...

# ---------- Decode helpers (§3.0) ----------

@torch.no_grad()
def greedy_decode(model, tok, prompt, max_new=384):
    ids = tok(prompt, return_tensors="pt",
              add_special_tokens=False).input_ids.to(model.device)
    t0 = time.perf_counter()
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    latency = time.perf_counter() - t0
    gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True)
    n_tokens = int(out.size(1) - ids.size(1))
    return gen, n_tokens, latency

@torch.no_grad()
def sample_decode(model, tok, prompt, K=5, T=0.7, top_p=0.95, max_new=384):
    ids = tok(prompt, return_tensors="pt",
              add_special_tokens=False).input_ids.to(model.device)
    t0 = time.perf_counter()
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=True,
        temperature=T, top_p=top_p,
        num_return_sequences=K,
        pad_token_id=tok.eos_token_id,
    )
    latency = time.perf_counter() - t0
    gens = [tok.decode(out[i, ids.size(1):], skip_special_tokens=True)
            for i in range(K)]
    n_tokens = int((out.size(1) - ids.size(1)) * K)
    return gens, n_tokens, latency

# ---------- One cell ----------

def run_cell(hf_id, tag, task, mode, test_records, out_path, log_path):
    torch.manual_seed(1234)
    model, tok = load_model(hf_id)
    fout = open(out_path, "w")
    n_correct = 0
    for i, rec in enumerate(test_records):
        sys_text, user_text = task_prompt(task, rec)
        prompt = build_chat_prompt(tok, hf_id, sys_text, user_text)

        row = {"id": rec.get("id", i), "task": task, "mode": mode,
               "top1_ok": False, "majority_ok": None, "any_ok": None,
               "top1_gen": None, "sample_gens": [],
               "n_gen_tokens": 0, "latency_s": 0.0}

        if mode == "fewshot":
            gen, ntok, lat = greedy_decode(model, tok, prompt)
            ok = score_one(task, gen, rec)
            row.update(top1_ok=bool(ok), top1_gen=gen[:600],
                       n_gen_tokens=ntok, latency_s=lat)
            n_correct += int(ok)

        elif mode == "sc":
            gens, ntok, lat = sample_decode(model, tok, prompt, K=5)
            oks  = [score_one(task, g, rec) for g in gens]
            keys = [extract_answer_key(task, g, rec) for g in gens]
            keys = [str(k) if k is not None else None for k in keys]
            cnt  = Counter(k for k in keys if k is not None)
            top_key = cnt.most_common(1)[0][0] if cnt else None
            majority_ok = any(o and k == top_key
                              for o, k in zip(oks, keys))
            row.update(majority_ok=bool(majority_ok),
                       any_ok=any(oks),
                       sample_gens=[g[:600] for g in gens],
                       n_gen_tokens=ntok, latency_s=lat)
            n_correct += int(majority_ok)

        elif mode == "tot":
            # adapter implements the §3.3/§4 interface
            from src.tot_adapters import build_adapter
            adapter = build_adapter(task, rec, hf_id, tok)
            t0 = time.perf_counter()
            best, ntok = tot_solve(adapter, model, tok)  # see §3.3
            lat = time.perf_counter() - t0
            ok = adapter.is_correct(best)
            row.update(top1_ok=bool(ok), top1_gen=best[:600],
                       n_gen_tokens=ntok, latency_s=lat)
            n_correct += int(ok)
        else:
            raise ValueError(mode)

        fout.write(json.dumps(row) + "\n"); fout.flush()
    fout.close()
    print(f"[{tag} {task} {mode}] {n_correct}/{len(test_records)}")

# ---------- CLI ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", required=True)
    ap.add_argument("--tag",   required=True)   # phi4_14b | qwen3_14b | ...
    ap.add_argument("--task",  required=True)   # g24 | pq | bw | ...
    ap.add_argument("--mode",  required=True)   # fewshot | sc | tot
    ap.add_argument("--data",  required=True)
    ap.add_argument("--limit", type=int, required=True)
    args = ap.parse_args()

    out  = f"results/multimodel_v2/{args.tag}_{args.task}_{args.mode}.jsonl"
    logp = f"logs/multimodel_v2/{args.tag}_{args.task}_{args.mode}.log"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(logp).parent.mkdir(parents=True, exist_ok=True)

    # Skip if complete (§6.4)
    if Path(out).exists() and sum(1 for _ in open(out)) >= args.limit:
        print(f"[skip] {out}")
        sys.exit(0)

    recs = [json.loads(l) for l in open(args.data)]
    recs = recs[: args.limit]
    run_cell(args.hf_id, args.tag, args.task, args.mode, recs, out, logp)
```

Driver loop (one process per cell, sized to your hardware):

```bash
# Pseudo-driver — fill in per-cell GPU pinning to suit your hardware.
MODELS=(
  "phi4_14b:microsoft/phi-4"
  "qwen3_14b:Qwen/Qwen3-14B"
  "gemma3_27b:google/gemma-3-27b-it"
  "llama3_70b:meta-llama/Llama-3.3-70B-Instruct"
)
TASKS=(
  "g24:data/24_test.jsonl:100"
  "pq:data/prontoqa_test.jsonl:100"
  "bw:data/blocksworld_test.jsonl:100"
  "gc:data/graphcolor_test.jsonl:100"
  "numpath:data/numpath_test.jsonl:200"
  "rulechain:data/rulechain_test.jsonl:200"
  "clutrr:data/clutrr_test.jsonl:200"
  "proofwriter:data/proofwriter_test.jsonl:200"
)
for mode in fewshot sc tot; do
  for entry_m in "${MODELS[@]}"; do
    IFS=':' read -r tag hf <<<"$entry_m"
    for entry_t in "${TASKS[@]}"; do
      IFS=':' read -r task data limit <<<"$entry_t"
      CUDA_VISIBLE_DEVICES=$NEXT_FREE_GPU python -m src.run_cell \
        --hf_id "$hf" --tag "$tag" --task "$task" --mode "$mode" \
        --data  "$data" --limit "$limit" \
        > "logs/multimodel_v2/${tag}_${task}_${mode}.log" 2>&1
      python -m src.update_summary \
        --cell "results/multimodel_v2/${tag}_${task}_${mode}.jsonl"
    done
  done
done
```

Replace `$NEXT_FREE_GPU` with your own scheduling logic (parsing
`nvidia-smi`, a fixed pool, a queue, etc.). Cells write to distinct
files, so any number of cells may run concurrently without contention,
provided their GPU memory budgets fit.
