# OOD eval datasets: ProntoQA + Blocksworld

We use these as **out-of-domain probes** for the LoRA trained on Game-of-24-varied. We **do not train** on either — they're eval-only. Goal: check if the LoRA's effect (catastrophic forgetting? task-agnostic transfer? z-channel utility?) shows up on radically different reasoning types.

This doc covers:
1. What each dataset looks like
2. The exact prompt we feed the model
3. Where the data comes from and how we sliced it
4. What ground-truth labels we use
5. Scoring rules
6. (Notes on what would change if we *did* want to train on these)

---

## 1. ProntoQA — synthetic deductive reasoning

**Source.** [renma/ProntoQA](https://huggingface.co/datasets/renma/ProntoQA) on HuggingFace Hub. 500 records, single split (`validation`). Originally from [Saparov & He, ICLR 2023](https://arxiv.org/abs/2210.01240).

### What a record looks like

Raw HuggingFace record (truncated):
```json
{
  "id": "ProntoQA_1",
  "answer": "B",
  "context": "Jompuses are not shy. Jompuses are yumpuses. Each yumpus is aggressive. Each yumpus is a dumpus. ... Max is a yumpus.",
  "options": ["A) True", "B) False"],
  "question": "Is the following statement true or false? Max is sour."
}
```

Each problem is a synthetic taxonomy — chains of subclass / property rules over made-up concept names (`yumpus`, `tumpus`, `jompus`...) — plus one statement to verify. The synthetic vocabulary is critical: it prevents the model from cheating with world knowledge.

### Our prompt format

3-shot prompt with the first 3 records used as fewshot exemplars (same template); the rest go to test. The instruction explicitly demands a single-letter answer:

```
You are given a set of facts and rules. Use them to determine whether the
statement at the end is true or false. Answer with exactly one letter:
'A' for true, 'B' for false. Do not add any explanation.

Context: <ex1.context>
<ex1.question>
A) True B) False
Answer: <ex1.answer>

Context: <ex2.context>
...
Answer: A

Context: <ex3.context>
...
Answer: A

Context: <test.context>
<test.question>
A) True B) False
Answer:
```

The full prompt is wrapped once more by Qwen's chat template (`<|im_start|>user … <|im_end|><|im_start|>assistant`) at eval time inside [src/eval_ood_generic.py](../src/eval_ood_generic.py).

### Test set construction

Driver: [data/prepare_ood_evals.py](../data/prepare_ood_evals.py).

- Records 0–2: **fewshot exemplars** baked into every prompt
- Records 3–202: **200 test records** → `data/prontoqa_test.jsonl`

Each line:
```json
{"id": "ProntoQA_4",
 "prompt": "<full 3-shot prompt as above>",
 "answer_label": "B"}
```

### Scoring

[src/score_ood.py](../src/score_ood.py) `score_prontoqa`:

1. Take the first non-empty line of the model's generation.
2. Look for an isolated `A` or `B` token (`\bA\b` or `\bB\b`) — exact match against `answer_label`.
3. Fallback: if neither letter, scan for "true" / "false" (case-insensitive) and map to A / B.

Strict letter-match means a model that explains its reasoning instead of answering with one letter gets graded on whether the explanation *starts* with A/B. This is intentional: instruction-following is part of the eval.

---

## 2. Blocksworld — natural-language symbolic planning

**Source.** [tasksource/planbench](https://huggingface.co/datasets/tasksource/planbench) on HuggingFace Hub, config `task_1_plan_generation`. 500 blocksworld records (filtered `domain == "blocksworld"`, `prompt_type == "oneshot"`). From [PlanBench, Valmeekam et al. 2022](https://arxiv.org/abs/2206.10498).

### What a record looks like

Raw HuggingFace record:
```json
{
  "task": "task_1_plan_generation",
  "prompt_type": "oneshot",
  "domain": "blocksworld",
  "instance_id": "...",
  "query": "<full natural-language prompt with 1 in-context example>",
  "ground_truth_plan": "(unstack blue orange)\n(stack blue yellow)\n..."
}
```

The `query` is **already** a complete one-shot prompt: it states the action set, lists the rules, gives one solved example, then poses the new problem with `[PLAN]` priming. We just pass `query` through Qwen's chat template; no further fewshot construction needed.

### Our prompt format

`query` verbatim, end of which looks like:

```
[STATEMENT]
As initial conditions I have that, the red block is clear, the orange block is clear,
the yellow block is clear, the hand is empty, the yellow block is on top of the blue
block, the red block is on the table, the blue block is on the table and the orange
block is on the table.
My goal is to have that the red block is on top of the yellow block, the blue block
is on top of the orange block and the yellow block is on top of the blue block.

My plan is as follows:

[PLAN]
unstack the yellow block from on top of the blue block
stack the yellow block on top of the red block
...
[PLAN END]

[STATEMENT]
As initial conditions I have that, ...
My goal is to have that ...

My plan is as follows:

[PLAN]
```

The model is expected to continue with action lines and `[PLAN END]`.

### Test set construction

- Filter: `domain == "blocksworld"` AND `prompt_type == "oneshot"` → 500 records
- Shuffle with seed=1234, take first **200** → `data/blocksworld_test.jsonl`

Schema:
```json
{"id": "<instance_id>",
 "prompt": "<full PlanBench query>",
 "answer_label": "(unstack blue orange)\n(stack blue yellow)\n(unstack orange red)\n(stack orange blue)"}
```

`answer_label` is the PlanBench `ground_truth_plan`, normalised to `(action arg1 arg2)` per line (PDDL-style).

### Scoring

[src/score_ood.py](../src/score_ood.py) `score_blocksworld`:

1. Extract every line in the generation that matches `^\([\w\- ]+\)\s*$` — i.e. the model's action list.
2. Stop at `[PLAN END]` or a new `[STATEMENT]` marker (model sometimes hallucinates more problems).
3. **Exact-list match** vs the ground-truth lines.

Exact-match is strict — semantically-valid alternative plans count as wrong. PlanBench's full evaluator runs Fast-Downward to verify any plan reaches the goal, but we deliberately use exact-match here because we lack a planner in the loop and want a fast, reproducible scorer for an OOD probe.

---

## 3. Why no training on these

The HypPlan pipeline trains on **arithmetic** trees (Game-of-24 variations + Countdown). The trees define `winning_ops(state, target)` — the set of one-step operations whose *resulting state* can still reach the target. This is the supervision signal for the Stage-1 head and the DAgger oracle in Stage 2.

For ProntoQA / Blocksworld, the analogous oracles would be:
- **ProntoQA**: `winning_inferences(state, query)` — set of rule applications that, when applied to the current fact set, makes a step toward proving (or disproving) the query. Buildable: BFS over rule applications, finite.
- **Blocksworld**: `winning_actions(state, goal)` — set of optimal-distance actions toward the goal. Requires a real planner (Fast-Downward / pyperplan).

Building either is **>1 day of careful work** per task plus tree enumeration / hidden-state caching / Stage-1 head training. See [README.md § OOD generalization probes](../README.md#ood-generalization-probes-no-head-training-prontoqa--blocksworld) for the full pipeline recipe and time estimate (~1.5 days ProntoQA, ~3–4 days Blocksworld).

For this iteration we **only evaluate**, with three conditions per dataset (base, LoRA-no-z, LoRA + random-z). Random-z is a noise control to test whether the LoRA's z-handling channel is even alive on these tasks.

---

## 4. End-to-end commands

```bash
# Build test sets (one-time; uses HuggingFace cached datasets)
python data/prepare_ood_evals.py

# Eval all 3 modes × 2 datasets in parallel (one GPU per condition).
# See README § "Three eval conditions" for the launch loop.

# Score
python -m src.score_ood --task prontoqa     --input results/eval_ood/prontoqa_base.jsonl
python -m src.score_ood --task prontoqa     --input results/eval_ood/prontoqa_lora.jsonl
python -m src.score_ood --task prontoqa     --input results/eval_ood/prontoqa_lora_randz.jsonl
python -m src.score_ood --task blocksworld  --input results/eval_ood/blocksworld_base.jsonl
python -m src.score_ood --task blocksworld  --input results/eval_ood/blocksworld_lora.jsonl
python -m src.score_ood --task blocksworld  --input results/eval_ood/blocksworld_lora_randz.jsonl
```
