# OOD eval datasets: ProntoQA + Blocksworld

We use these as **out-of-domain probes** for the LoRA trained on Game-of-24-varied. The LoRA itself is **never** trained on these tasks — only the Stage-1 hyperbolic head is, so that we have a meaningful task-specific `z` to inject at eval time.

This doc covers:
1. What each dataset looks like
2. The exact prompt we feed the model at eval
3. Where the data comes from and how we sliced it
4. What ground-truth labels we use
5. Scoring rules
6. **Stage-1 head training pipeline** for each task (oracle → tree → cache → head)

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

## 3. Stage-1 head training pipeline

Same Stage-1 recipe as Game-of-24 / Countdown: enumerate trees → cache hidden states → train origin-ranking head. The differences are the **oracle** (what counts as a "winning" next step) and the **state-rendering** (how each tree node is converted to text the LLM sees).

### 3.1 ProntoQA Stage-1

**Oracle** ([src/oracle_pronto.py](../src/oracle_pronto.py)): forward-chaining deduction.
- *State* = frozenset of `(predicate, bool)` facts known about the entity (Max / Sam / etc.). Each problem has a single entity and a single starting fact.
- *Action* = apply one Horn-clause rule whose premise matches a known fact, deriving a new fact.
- *Decidable* = the query predicate appears in the state with either value (so we know if the query is true or false).
- `enumerate_tree(problem)` does a BFS over reachable states, marks decidable ones, then back-propagates BFS distance as `v_value` (how many rule applications away from a decidable state).

Trees per problem: ~90–760 nodes, max depth ~14, v-value range 0–6.

**State rendering** matches the eval prompt's deductive style:

```
Initial fact: Max is a yumpus.
Derived so far:
  Max is aggressive.
  Max is a dumpus.
  Max is red.
Question: is Max sour?
```

**Train/val/test split** ([data/generate_tree_data_pronto.py](../data/generate_tree_data_pronto.py)): seed-1234 shuffle of the 500 records, then 250 / 50 / 200 train / val / test. The 200 test indices are **the same** as our eval test set — we never train on them.

**Tree caching**: forward each rendered state's last-token last-layer hidden state through frozen `Qwen/Qwen2.5-14B-Instruct` (bf16). Output: `data/pronto_trees_qwen14b/{train,val,test}/problem_<idx>.{pt,_npy}`. Sharded across 4 GPUs, takes ~25 min total.

**Head config**: [configs/head_pronto_qwen14b_rank.yaml](../configs/head_pronto_qwen14b_rank.yaml) — 32-d Poincaré, origin-ranking margin loss, 20 epochs, lr 1e-3.

**Output**: `checkpoints/head_pronto_qwen14b_rank/head.pt`, dimensionally compatible with the existing `up_projector.pt` of the z-arm-balanced LoRA — so we can swap heads at eval time with `--head_override`.

### 3.2 Blocksworld Stage-1

**Oracle** ([src/oracle_blocksworld.py](../src/oracle_blocksworld.py)): standard 4-op blocksworld.
- *State* = frozenset of facts: `("on", X, Y)`, `("ontable", X)`, `("clear", X)`, `("holding", X)`, `("handempty",)`.
- *Actions*: `pick-up`, `put-down`, `stack`, `unstack` — each with the standard preconditions / effects from the PDDL domain.
- `parse_problem(prompt)` extracts initial state and goal from the trailing `[STATEMENT]` block of a PlanBench query.
- `enumerate_tree(problem, max_nodes=2000)` BFS over reachable states, marks goal-reaching ones. v-value = BFS distance to a goal state (over undirected parent↔child edges).

Trees per problem: 100–2000 nodes (capped); v-value range 0–~12.

**State rendering**:

```
Current state:
  the hand is empty
  the orange block is on top of the blue block
  the red block is on top of the orange block
  the blue block is on the table
  the red block is clear
Goal:
  blue on yellow
  orange on blue
```

**Train/val/test split** ([data/generate_tree_data_blocksworld.py](../data/generate_tree_data_blocksworld.py)): same shuffled 500 oneshot blocksworld records — records 0–199 = test (held out from head training), 200–449 = train, 450–499 = val. Sharded across 4 GPUs, takes ~25 min.

**Head config**: [configs/head_blocksworld_qwen14b_rank.yaml](../configs/head_blocksworld_qwen14b_rank.yaml) — same architecture (32-d Poincaré, origin-ranking).

**Output**: `checkpoints/head_blocksworld_qwen14b_rank/head.pt`.

### 3.3 What's different vs Game-of-24

| | Game-24 / Countdown | ProntoQA | Blocksworld |
|---|---|---|---|
| State | tuple of remaining numbers | frozenset of (predicate, bool) facts | frozenset of (on / ontable / clear / holding / handempty) facts |
| Action | (a, op, b) → r | apply rule R: P(x,V) ⇒ Q(x,V′) | pick-up / put-down / stack / unstack |
| Success | remaining = [target] | query predicate is in state | goal facts ⊆ state |
| Branching | 4 ops × N(N−1)/2 operand pairs | # rules whose premise matches | # actions whose preconds hold |
| Avg tree size | 50–500 | 90–760 | 100–2000 |

The Stage-1 *training infrastructure* ([src/train_head.py](../src/train_head.py)) is unchanged — it just reads `problem_*.pt` + `hidden_*.npy` and trains the origin-ranking margin head. We reuse the existing `TreeCacheDataset`.

---

## 4. End-to-end commands

```bash
# 4.1 — Build OOD test sets (eval-only)
python data/prepare_ood_evals.py

# 4.2 — Build Stage-1 tree caches (sharded across 4 GPUs)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i+1)) python data/generate_tree_data_pronto.py \
    --shard_rank $i --shard_world 4 &
done; wait
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i+1)) python data/generate_tree_data_blocksworld.py \
    --shard_rank $i --shard_world 4 --splits train,val &
done; wait

# 4.3 — Train Stage-1 heads (single GPU each, ~30 min)
HYPPLAN_DIST_BACKEND=gloo CUDA_VISIBLE_DEVICES=6 \
  python -m src.train_head --config configs/head_pronto_qwen14b_rank.yaml
HYPPLAN_DIST_BACKEND=gloo CUDA_VISIBLE_DEVICES=7 \
  python -m src.train_head --config configs/head_blocksworld_qwen14b_rank.yaml

# 4.4 — Eval (4 modes per task: base / lora / lora_randz / lora_taskz)
# `lora_taskz` will use --head_override pointing at the new task-specific head.
# (Eval driver update + final results table forthcoming.)

# Scoring (already supports both tasks)
python -m src.score_ood --task prontoqa    --input results/eval_ood/prontoqa_*.jsonl
python -m src.score_ood --task blocksworld --input results/eval_ood/blocksworld_*.jsonl
```

### Time budget — actual vs original estimate

| Step | Original estimate | Actual |
|---|---|---|
| ProntoQA oracle + tree gen + head | 1.5 days | ~1 hour (no planner needed; Horn-clause forward chaining is trivial) |
| Blocksworld oracle + tree gen + head | 3–4 days | ~1 hour (built minimal Python BFS for the 4-op domain instead of using Fast-Downward) |

The original README estimate assumed we'd integrate a real PDDL planner and write proper task-rendering on par with Game-24's; in practice, a minimal hand-written BFS oracle was enough to populate trees with sensible v-values, since even un-optimal-distance v-values give a margin-ranking head useful supervision (state nearer to goal/decidable should have smaller `|z|`).
