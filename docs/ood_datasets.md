# OOD eval datasets: ProntoQA + Blocksworld + Graph Coloring

We use these as **out-of-domain probes** for the LoRA trained on Game-of-24-varied. Two probe modes:

1. **HypPlan**: the G24-varied LoRA + a task-specific Stage-1 hyperbolic head. The LoRA itself is **never** trained on these tasks — only the Stage-1 head is, so that we have a meaningful task-specific `z` to inject at eval time.
2. **Planning-Tokens (PT) baseline**: a separate Qwen2.5-14B + LoRA SFT'd on each task's own training data with discrete operator/action planning tokens prepended to each step. Different setup from HypPlan (in-domain training, not OOD transfer), used as a reference baseline.

This doc covers:
1. What each dataset looks like
2. The exact prompt we feed the model at eval
3. Where the data comes from and how we sliced it
4. What ground-truth labels we use
5. Scoring rules (including CD strict vs lenient + BW exact-match vs goal-reaching)
6. **Stage-1 head training pipeline** for each task (oracle → tree → cache → head)
7. **Planning-Tokens SFT baseline** — train Qwen2.5-14B on each task's planning-token-augmented data and eval

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

We compute **two** metrics on the model's output:

#### Metric A: exact-match (strict, fast, often misleading)

[src/score_ood.py](../src/score_ood.py) `score_blocksworld`:

1. Extract every action line: `(action arg1 ...)` or its natural-language equivalent (the in-context example uses NL phrasing like "unstack the yellow block from on top of the blue block"; we normalize both to PDDL form).
2. Stop at `[PLAN END]` or a new `[STATEMENT]` marker (model sometimes hallucinates more problems).
3. Compare the extracted Python list to the gold list with `==`.

This is what we reported as "1%/0%/2%/2%" — those numbers **drastically under-estimate** correctness because:
- The model may emit the gold plan AND keep generating extra actions afterward → exact-match fails even though the goal IS reached at step 4.
- Multiple optimal plans may exist; picking a different valid one → exact-match fails.

#### Metric B: goal-reaching (proper Blocksworld metric)

[src/score_ood.py](../src/score_ood.py) `score_blocksworld_goal_reaching`:

1. Parse the puzzle's initial state and goal facts from the prompt ([src/oracle_blocksworld.py](../src/oracle_blocksworld.py) `parse_problem`).
2. Simulate the model's predicted action sequence forward from the initial state, applying preconditions/effects of each action.
3. After **each** applied action, check whether `goal_facts ⊆ current_state`.
4. **Correct** iff the goal is achieved at any step (and we stop at the first illegal action — i.e. one whose preconditions don't hold).

This matches the spirit of PlanBench's Fast-Downward verification — alternative valid plans count, and the model is rewarded for finding a working plan even if it goes on to spam extra actions.

#### Concrete example (real test case from base mode)

```
Puzzle: get   blue on yellow,   orange on blue
Initial:      blue on orange,   orange on red,   yellow on table

GOLD plan (4 actions):                   MODEL plan (8 actions, extracted):
  (unstack blue orange)                    (unstack blue orange)         ✓
  (stack blue yellow)                      (stack blue yellow)           ✓
  (unstack orange red)                     (unstack orange red)          ✓
  (stack orange blue)                      (stack orange blue)  ← goal! ✓
                                           (unstack blue orange)  ← extra
                                           (stack blue yellow)
                                           (unstack orange red)
                                           (stack orange blue)
```
- Metric A (exact-match): **WRONG** (8 actions ≠ 4 actions)
- Metric B (goal-reaching): **CORRECT** (goal achieved at step 4)

We report **both** metrics; goal-reaching is the meaningful one.

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

### 3.2 Graph Coloring Stage-1

**Task**: 3-coloring on 5- or 6-vertex graphs with edge density ∈ {0.2, 0.3, 0.4, 0.5, 0.6}. Stratified across (n, density), 500 problems total (250/50/200 train/val/test).

**Oracle** ([src/oracle_graphcolor.py](../src/oracle_graphcolor.py)): standard CSP search.
- *State* = tuple of (vertex_id, color) for already-colored vertices, in canonical order.
- *Action* = pick the next uncolored vertex (lowest unfilled id) + assign one of {R, G, B} that doesn't conflict with already-colored neighbors.
- *Complete* = all n vertices colored consistently.
- v_value = BFS distance from each state to a complete leaf (over undirected parent↔child edges).

Trees per problem: 40–256 nodes, max v_value = n.

**State rendering**:

```
Graph: 5 vertices (V0, V1, V2, V3, V4)
Edges: (V0,V1), (V1,V2), (V2,V3)
Colored so far:
  V0 = red
  V1 = green
Uncolored: V2, V3, V4
```

**Train/val/test split** ([data/generate_tree_data_graphcolor.py](../data/generate_tree_data_graphcolor.py)): all 500 problems generated by [data/prepare_graphcolor.py](../data/prepare_graphcolor.py); first 250 = train (head training), 250-299 = val, 300-499 = test (held out from head training; same 200 used in eval).

**Tree caching**: forward each rendered state through frozen Qwen2.5-14B (bf16). Output: `data/graphcolor_trees_qwen14b/{train,val,test}/`. Sharded across 4 GPUs, ~3 min total (small trees).

**Head config**: [configs/head_graphcolor_qwen14b_rank.yaml](../configs/head_graphcolor_qwen14b_rank.yaml).

**Output**: `checkpoints/head_graphcolor_qwen14b_rank/head.pt`.

**Eval prompt** is a 2-shot fewshot built by `data/prepare_graphcolor.py` showing the model how to format `V<i> = <color>` outputs:

```
You are given a graph 3-coloring task. Output one assignment per line in
the form 'V<i> = <color>' with color in {R, G, B}, then stop.

Graph 3-coloring task.
Vertices: V0, V1, V2, V3
Edges: (V0,V1), (V1,V2), (V2,V3)
...
Coloring:
V0 = red
V1 = green
V2 = red
V3 = green
Done.

[next example...]

Graph 3-coloring task.
Vertices: V0, V1, V2, V3, V4, V5
Edges: ...
Coloring:
```

### 3.3 Blocksworld Stage-1

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

### 3.4 What's different vs Game-of-24

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

# Scoring
python -m src.score_ood --task prontoqa            --input results/eval_ood/prontoqa_*.jsonl
python -m src.score_ood --task blocksworld         --input results/eval_ood/blocksworld_*.jsonl   # exact-match (under-estimate)
python -m src.score_ood --task blocksworld_goal    --input results/eval_ood/blocksworld_*.jsonl   # goal-reaching (proper)
```

### Time budget — actual vs original estimate

| Step | Original estimate | Actual |
|---|---|---|
| ProntoQA oracle + tree gen + head | 1.5 days | ~1 hour (no planner needed; Horn-clause forward chaining is trivial) |
| Blocksworld oracle + tree gen + head | 3–4 days | ~1 hour (built minimal Python BFS for the 4-op domain instead of using Fast-Downward) |

The original README estimate assumed we'd integrate a real PDDL planner and write proper task-rendering on par with Game-24's; in practice, a minimal hand-written BFS oracle was enough to populate trees with sensible v-values, since even un-optimal-distance v-values give a margin-ranking head useful supervision (state nearer to goal/decidable should have smaller `|z|`).

---

## 5. Planning-Tokens (PT) SFT baseline

To compare HypPlan against [Planning Tokens (Wang et al. 2023)](https://arxiv.org/abs/2310.05707) on the same 3 OOD probes, we SFT a Qwen2.5-14B LoRA on each task's training data with operator/action tokens inserted before each reasoning step.

### Phi-1.5 GSM8K verification first

Before running on OOD tasks, we verified our PT implementation against the paper's reported numbers. Phi-1.5 + GSM8K, LoRA r=16, 10 epochs:
- Paper: baseline 12.5% → arithmetic-PT 15.0% (+2.5pp)
- Ours: baseline 30.4% → arithmetic-PT 32.1% (+1.7pp, on a 355-record sample)

Trend matches. Higher absolute numbers are due to slight prompt format differences. See [src/train_sft_gsm8k.py](../src/train_sft_gsm8k.py), [src/eval_gsm8k.py](../src/eval_gsm8k.py).

### Per-task PT vocabulary + training data

Built by [data/prepare_pt_ood_data.py](../data/prepare_pt_ood_data.py):

| Task | Planning tokens | Train data source |
|---|---|---|
| **CD** | `<PLAN:+>`, `<PLAN:->`, `<PLAN:*>`, `<PLAN:/>`, `<PLAN:ANS>` | Existing CD trajectories `data/cd_train_sft.jsonl` (1000 records) |
| **BW** | `<PLAN:PICKUP>`, `<PLAN:PUTDOWN>`, `<PLAN:STACK>`, `<PLAN:UNSTACK>`, `<PLAN:ANS>` | PlanBench gold plans (records 200-449, 250 records) |
| **PQ** | `<PLAN:DERIVE_TRUE>`, `<PLAN:DERIVE_FALSE>`, `<PLAN:ANS>` | Oracle-generated forward-chaining proofs (records 0-249, 250 records) |
| **GC** | `<PLAN:R>`, `<PLAN:G>`, `<PLAN:B>`, `<PLAN:ANS>` | Gold colorings from `_is_3_colorable` backtracking (250 records, generated alongside test set) |

For PQ we generate proofs from `src/oracle_pronto.py`'s forward chaining (greedy first-applicable rule until query is decidable).

### Training (Qwen2.5-14B, LoRA r=16)

Configs: `configs/sft_pt_{cd,bw,pq}_qwen14b.yaml`. Driver: [src/train_sft_gsm8k.py](../src/train_sft_gsm8k.py) (DDP-2 gloo). 3 SFT runs in parallel, 2 GPUs each (6 GPUs total, ~7 min wall for all three).

### Eval

Driver: [src/eval_pt_ood.py](../src/eval_pt_ood.py). Builds `Question: <q>\nAnswer:` prompts matching the SFT format and decodes greedy. Per-task scoring uses the same scorers as HypPlan eval, so results are directly comparable.

### Scoring details for CD (strict vs lenient)

CD generations look syntactically clean — model emits `<PLAN:OP> Step N: a op b = r. ...` lines and finishes with `<PLAN:ANS> Answer: T`. Two scorers:

- **Lenient** (string match): does `Answer:\s*N` appear in the output with `N == target`? → 58/100 = 58%
- **Strict** (`src/evaluate_generic.parse_and_validate_generic`): also verifies (a) each step's arithmetic is correct, (b) only numbers from the current `remaining` pool are used as operands. → **0/100 = 0%**

The 58pp gap reveals that the PT-SFT model is **hallucinating numbers**:

```
pool=[5, 8, 8, 9, 9, 75], target=647
gen: <PLAN:*> Step 1: 5 * 8 = 40. Remaining: 8 9 9 40 75
     <PLAN:+> Step 2: 8 + 9 = 17. Remaining: 9 17 40 75
     <PLAN:-> Step 3: 75 - 17 = 58. Remaining: 9 40 58
     <PLAN:*> Step 4: 9 * 58 = 522. Remaining: 40 522
     <PLAN:+> Step 5: 40 + 522 = 562. Remaining: 562
     <PLAN:+> Step 6: 562 + 85 = 647.   ← 85 is NOT in remaining!
     <PLAN:ANS> Answer: 647
```

The model writes "Answer: 647" (matches gold) but step 6 used the number 85 which was never in the pool. Lenient says correct, strict says wrong. We report strict.

### Final results

Numbers live in the consolidated **Headline results** in [README.md](../README.md). Per-task interpretation:

- **PQ** — *format learning, not reasoning*. PT-SFT regresses below base because the model overfits to "emit Answer: A by default" given the small training set + biased label distribution. Short generations like `<PLAN:DERIVE_TRUE> Step 1: ... Answer: A` regardless of question content.
- **BW** — *memorization*. 94.5% is misleading: PT-SFT trains on 250 records of PlanBench's gold plans and the test set comes from the same distribution. The model regurgitates structurally similar plans that mostly happen to reach the goal. Not evidence of compositional planning.
- **GC** — *PT-SFT loses to HypPlan-no-z* (64% vs 68%). PT-SFT correctly emits `<PLAN:COLOR>` tags and assigns vertex colors, but **doesn't always check edge constraints** — example: model outputs `V3=B, V4=B` for adjacent vertices V3-V4. The G24-LoRA has stronger constraint-checking habits learned from its arithmetic training.
- **CD** — *hallucinated arithmetic*. Goal achieved iff "Answer: T" string appears (lenient = 58%); proper validation drops it to 0%. Don't trust the lenient number.

### Bottom-line comparison vs HypPlan

PT-SFT is task-specific in-domain training, fundamentally different from HypPlan's G24-only training + OOD probes. The 4-task picture:
- **HypPlan transfer is real on G24-similar tasks** (graph coloring +7pp over base; constraint satisfaction with sequential decisions resembles G24's "pick op + check constraint" structure).
- **PT-SFT either underperforms (PQ, GC) or memorizes (BW)**. The best-case PT-SFT (BW 94.5%) is a memorization ceiling, not generalization.
- **The geometric z signal still doesn't transfer** — across all 4 tasks, `lora + task-z ≤ lora + no-z`. The negative result on the head's z is robust.

---

## 6. HypPlan Stage-1+2 trained IN-DOMAIN per task

The OOD probes above kept the LoRA frozen on G24-varied and only swapped Stage-1 heads. To test the full methodology, we trained **a fresh Stage-2 LoRA + UpProjector on each OOD task's own training data** with that task's Stage-1 head.

### 6.1 Setup

- **Trainer**: [src/train_stage2_dagger_ood.py](../src/train_stage2_dagger_ood.py) — generic DAgger trainer that dispatches by task to per-task adapters.
- **Adapters**: [src/dagger_ood_adapters.py](../src/dagger_ood_adapters.py) — `ProntoQAAdapter`, `BlocksworldAdapter`, `GraphColorAdapter`. Each implements `winning_steps(state)`, `parse_step`, `validate_apply`, `apply_step`, `is_solved`, `render_state`, `make_prompt`, etc.
- **Rollout**: [src/dagger_rollout_ood.py](../src/dagger_rollout_ood.py) — generic z-injection rollout consuming an adapter.
- **Eval**: [src/eval_stage2_indomain.py](../src/eval_stage2_indomain.py) — reuses `rollout_one` at temp=0 with a custom step-by-step prompt matched to the training prompt.
- **Configs**: `configs/stage2_dagger_{pq,bw,gc}_qwen14b.yaml`.
- **Training prompt**: a custom chat-template prompt asking for `Step 1: …\nStep 2: …\n…\nAnswer: <X>`. Same prompt at train + eval (clean A/B for the methodology).

### 6.2 Versioning of the in-domain Stage-1+2 result

Cross-baseline numbers are in the consolidated **Headline results** in [README.md](../README.md). This section tracks the per-version progression as we fixed bugs and tuned the trainer:

| Task | v1 (initial DAgger) | v2 (cyclic-pad fix) | v3 (3 rollouts × 4 epochs) |
|---|---|---|---|
| PQ | **75** | 75 | (unchanged) |
| BW (goal) | 0 | **10** | TBD |
| GC | **88** | 88 | (unchanged) |

PQ + GC: clear in-domain wins from v1 onward (+15pp and +27pp over base). BW: this section explains why v1 failed at 0%, how v2 lifted it to 10%, and what v3 is trying.

### 6.3 Why BW failed in v1 (0/100 solved)

Inspecting failed rollouts on the first version of the trained Stage-2 LoRA:

```
id=334 stopped=budget n_boundaries=16
gen: Step 1: unstack the blue block from on top of the orange block.   ← correct (matches gold)
     Step 2: put down the blue block.                                  ← wrong
     Step 3: pick up the blue block.                                   ← back to step-1's state
     Step 4: put down the blue block.
     Step 5: pick up the blue block.
     ... (cycles for 16 steps until depth budget) ...
```

**Diagnosis: the LoRA learned step 1 but not step 2+.** The model emits the correct first action (which is usually "unstack the topmost block" — a near-deterministic pattern across BW initial states), then falls back to alternating pick-up/put-down. 93/100 trajectories cycled until `max_steps=16`.

**Root cause: training-data sparsity from DDP sync.** Different ranks finished their rollout phases with different numbers of `(state, gold-step)` pairs (because BW rollouts have variable plan lengths and variable failure depths). To keep gloo's all-reduce calls aligned across ranks, the original sync truncated every rank's pair list to the **global minimum** count. Concretely:

- Each rank rolled out ~42 problems.
- Rank 0 (often rolling more successfully) collected 81 pairs.
- The slowest rank had 41 pairs.
- After truncation: every rank trained on 41 unique pairs/epoch × 6 ranks = ~246 unique pairs — but most were step-1 boundaries because rollouts failed at step 2.

Comparison across tasks:

| | per-record gold trajectory | rollout failure point | training pairs/epoch (v1) |
|---|---|---|---|
| PQ | 5–9 derivation steps | mostly succeeds (rules are listed in the prompt; pattern-match) | 184 |
| GC | 5–8 vertex assignments | mostly succeeds (3-color CSP is short and constrained) | 70 |
| BW | 4–12 actions | fails by step 2 most of the time | 41 |

So BW had both the longest gold trajectories *and* the shortest model rollouts, yielding the least training data for the most demanding task.

### 6.4 v2: cyclic-pad-to-global-max (0% → 10%)

We replaced the global-min truncation with **cyclic padding to global-max**: each rank repeats its own pairs (cyclically) to match the busiest rank's count. All ranks now make the same number of all-reduce calls (no hang), and no rollout data is wasted. Implemented in [src/train_stage2_dagger_ood.py:323](../src/train_stage2_dagger_ood.py#L323).

Effect on BW training:

| | v1 (global-min) | v2 (cyclic-pad) |
|---|---|---|
| Train steps/epoch | 82 | 266 (3.2×) |
| Final loss | 0.09 | 0.06 |
| **Eval correct** | 0/100 | **10/100** |
| `budget` (cycling) | 93 | 72 |
| `empty_oracle` (state not in oracle's BFS tree) | 7 | 17 |
| `solved` | 0 | 10 |

The fix lifts the methodology from broken to barely working on BW. PQ and GC numbers stay identical (their training was less starved by the truncation).

Bonus fix shipped at the same time: `optimizer.zero_grad(set_to_none=False)` plus a one-time grad-zero init before the first iteration. Without this, ranks whose loss returns `None` (empty winners) would have `p.grad = None` and skip `all_reduce`, hanging the others. Same bug class as global-min, different surface.

### 6.5 v3: more rollouts per problem × more epochs (currently running)

What still goes wrong in v2: 72/100 trajectories still hit the depth budget, and 17/100 wander into states the oracle's BFS tree (capped at 8000 nodes) never enumerated. So even when we keep all the rollout data, each problem only contributes ~4 boundaries because the rollout fails by step 4.

v3 adds three independent levers:

1. **`rollouts_per_problem: 3`** — added to the trainer. With `temperature=0.7`, three rollouts of the same problem explore different action choices, so we pick up boundaries at states that one greedy-ish rollout would never reach.
2. **`epochs: 4`** (was 2) — DAgger benefits from re-rolling under the *updated* policy. Once the LoRA learns step 2 in epoch 0, epoch 1's rollouts get further before failing, generating step-3 boundaries; etc.
3. **BW oracle `max_nodes`: 8000 → 30000** — covers states the model wanders into. Reduces `empty_oracle` from 17/100.

Expected combined effect: 6× more rollouts (3 per problem × twice as many epochs vs v2's 1×2), plus broader oracle coverage. Training time: roughly 6× longer than v2 (~50–60 min on 6 GPUs).

### 6.6 Open avenues for further BW improvement

If v3 still falls short of the base model's 41% goal-reaching, the next things to try:

- **Gold-trajectory bootstrap**: in addition to model rollouts, walk through the gold plan step-by-step and add `(gold_state_k, gold_step_{k+1})` pairs for every step. This gives the LoRA supervision on deep states it never reaches under its own policy. Strictly speaking this is "behavior cloning + on-policy correction" rather than vanilla DAgger, but it is a standard DAgger-with-warm-start variant.
- **Increase BW training set** beyond the current 250 PlanBench oneshot records. PlanBench has more BW instances available; we filtered to oneshot for prompt-format consistency.
- **Switch base to a planning-finetuned model** for BW only. Qwen-2.5-14B-Instruct has weak prior on BW planning; a base that's seen more PDDL or symbolic planning would give better step-1 → step-2 generalization.
- **Cycle detection in eval rollout**: terminate trajectories that revisit a state. This won't raise correctness but cleans up the `budget` failure mode and frees compute for re-rolls under a different sampling temperature.

The PQ + GC results confirm the methodology works. BW remains the hardest of the three because its state space is genuinely larger and the gold trajectories are longer — both of which amplify any training-data shortage.
