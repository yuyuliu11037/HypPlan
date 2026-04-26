# HypPlan: Tree-Distortion Hyperbolic Planning for LLM Reasoning

A two-stage framework. **Stage 1** teaches a small head to embed reasoning-tree states into a hyperbolic space so that `|z|` tracks solution-proximity — pure geometric supervision on an enumerated state tree, no language-model loss. **Stage 2** trains a fresh LoRA + `UpProjector` on top of a frozen base LLM using **DAgger with a tree oracle**: the current policy rolls out trajectories freely, the oracle labels winning ops at each reached state, and CE trains the LoRA on those labels. The frozen head's geometric `z` is injected as a virtual token before each step boundary.

Runs on **Game of 24**; the Countdown port (N=6 pool, variable integer target) is in progress. The MATH pipeline from the original HypPlan also lives in this tree (see `src/train_stage1.py` etc.) but the active pipeline is the two-stage Game-of-24 flow documented below.

Active base model: **Qwen-2.5-14B-Instruct** (hidden_dim=5120, frozen, used with a 3-shot chat prompt). Earlier Llama-3.1-8B-based runs are archived at the bottom of this README for reference.

---

## The two stages

### Stage 1 — hyperbolic head (LLM frozen)

For each Game-of-24 problem we enumerate the full state tree (root = initial 4 numbers, children = all legal (a, op, b) applications, leaves = 1-number terminal states). Each node's state text is encoded by the frozen base LLM. A small head MLP maps that hidden vector to a low-dim point (default `hyp_dim=32`) in a hyperbolic space via exp-map at the origin.

**Loss: `origin_ranking`.** A margin hinge on distance-to-origin, with target `v(s)` = BFS edge distance from state `s` to the nearest success leaf in the enumerated tree. For any sampled pair `(s_i, s_j)` with `v(s_i) < v(s_j)`:

`L = max(0, d_H(z_i, 0) − d_H(z_j, 0) + margin)`

This makes `|z|` track solution-proximity: states closer to a solution are pulled toward the origin, states farther are pushed outward. Supported manifolds: Poincaré ball and Lorentz hyperboloid (plus a Euclidean variant reserved for the geometry ablation). For Countdown's wide-range continuous v, a scale-invariant rank-based variant `origin_ranking_rank` is also supported.

### Stage 2 — DAgger with tree oracle (base and head frozen)

A fresh LoRA adapter on the frozen base + a small `UpProjector` (lifts the 32-dim hyperbolic point back to the model's hidden_dim) are the only trainable parts. Training uses **DAgger** (expert iteration, AlphaGo-style):

At each epoch, for each training problem:
1. **Rollout under current policy** — generate step-by-step with T=0.7, top-p=0.95, injecting `z_t` as a virtual token at each step boundary (z run only; the no-z control run skips the injection). Continue until a valid solution, an invalid step, or step budget exhausted.
2. **Oracle labeling** — for each step-boundary state reached, query the oracle (memoized recursive search via `src/oracle_24.py`): given `remaining`, return all ops whose resulting state can still reach 24.
3. **Invalid-step handling** — if the model emits a step with wrong arithmetic or hallucinated operands, truncate the trajectory at that step. Earlier valid states still contribute.
4. **Training pass** — for each collected (state, z, winning_ops) tuple, pick one winner (lex tiebreak) and CE-train the model to emit its full step text. Backprop into LoRA + UpProjector; head and base stay frozen.

The canonical state text for each boundary passes through frozen base + frozen head → `z`; up-projector produces `z_inj` injected before the next step's tokens. Loss is single-winner CE (phase 1); phase-2 upgrade would be log-of-sum over all winners' step-text likelihoods.

---

## Task-agnostic variant (Stage 2-varied)

The base pipeline above trains the LoRA on one fixed task: four numbers, target 24. That's what we measured (0.57 DAgger noz-stable, 0.55 z-stable). But the LoRA might have memorized "always reach 24" rather than learned a general skill — *use the z signal to steer reasoning*.

To test the weaker, more defensible claim — *given a meaningful z for any task, the LoRA reasons better* — we built a variant where the LoRA never sees a fixed target.

### Varied-target data

For each original Game-24 problem, we enumerate the full tree as before. Then we **sample (pool, target) pairs**:

- The pool is any internal node's `remaining` list (2, 3, or 4 integers).
- The target is the value of any size-1 terminal reachable from that node.
- Only integer intermediate states are kept (fractional subtrees are dropped).

This gives us ~20k unique pairs split 16,464 / 1,751 / 1,707 train / val / test ([data/24_varied_{train,val,test}.jsonl](data/)). Depth distribution per split: ~40% depth-3 (4-number pool), ~36% depth-2 (3-number), ~20% depth-1 (2-number), plus 24 trivial depth-0 cases in train ([data/generate_24_varied.py](data/generate_24_varied.py)).

The LoRA now sees targets ranging from 1 to 900+ across diverse pools. It can't memorize "reach 24" — it must read the target from the prompt each time.

### Generic prompt format

Instead of `Problem: 2 3 4 12` (implicit target 24), every example uses:

```
Numbers: 2 3 4 12 | Target: 24
Step 1: ...
```

The instruction is task-free: "Combine the given numbers using +, -, *, / to reach the target." No "24" baked in. Same `Step N: a op b = r` format as before, so the parser and oracle stay unchanged ([src/prompt_builders.py:fewshot_chat_prompt_generic](src/prompt_builders.py)).

### Stage 1 head (varied trees)

Head architecture is unchanged (Poincaré, hyp_dim=32, origin_ranking loss). The state text fed to the frozen base includes the target in the header, so the head learns a target-aware geometric embedding.

Trained on 16k varied trees for 20 epochs (~46 min on 1 GPU). Held-out eval:

| Split | Spearman(`d_origin`, v) | Rank accuracy |
|---|---|---|
| val | 0.452 | 80.8% |
| test | 0.453 | 80.1% |

"Rank accuracy" = given two nodes where one is closer to success, the head puts that one closer to origin. No val/test gap means the head generalizes across varied targets, not just memorizes ([checkpoints/head_24_varied_qwen14b_rank/](checkpoints/)).

### Stage 2 DAgger (varied-target)

Same DAgger loop as the base pipeline, but:

- Training problems are `(pool, target)` pairs, not 4-number strings.
- Oracle takes target as a parameter ([src/oracle_24_varied.py](src/oracle_24_varied.py)).
- Rollout and state rendering use generic format ([src/dagger_rollout_varied.py](src/dagger_rollout_varied.py)).
- Trainer: [src/train_stage2_dagger_varied.py](src/train_stage2_dagger_varied.py).

Currently running: 5000 train pairs, 2 epochs, 4-GPU DDP, Qwen-2.5-14B-Instruct base + Poincaré head.

### OOD evaluation plan (Phase 5)

The whole point is to check whether the task-agnostic LoRA, trained only on varied Game-24, can use a head trained on a *different* task. Countdown (6-number pool, variable 3-digit target) is the test.

Four conditions on the 100-problem Countdown test set, all using the **same** LoRA from above:

| Condition | Head used | What it tests |
|---|---|---|
| no-z | none (no injection) | LoRA alone on OOD task |
| rand-z | norm-matched Gaussian | does *any* vector at boundaries help? |
| **cd-head-z** | CD-trained head | **main test**: does the LoRA use task-specific geometric signal? |
| g24-head-z-on-cd | varied-G24 head applied to CD states | wrong geometry on right task — sanity control |

Plus one in-distribution sanity run: same LoRA on varied-G24 test (should beat the same-LoRA no-z baseline).

**Success criteria:** `cd-head-z > no-z`, `cd-head-z > rand-z`, and `cd-head-z > g24-head-z-on-cd` — all on the same LoRA. If so, the LoRA has learned to *use* geometric guidance in general, not just the specific guidance it trained with.

---

## OOD generalization probes: ProntoQA + Blocksworld + Graph Coloring

To probe whether the G24-trained LoRA transfers to OOD tasks, we run three test sets covering radically different reasoning types:

- **ProntoQA** ([renma/ProntoQA](https://huggingface.co/datasets/renma/ProntoQA), 200 records sampled): synthetic deductive reasoning. Given facts and rules, decide whether a statement is True (A) or False (B).
- **Blocksworld** ([tasksource/planbench](https://huggingface.co/datasets/tasksource/planbench), `task_1_plan_generation`, 200 blocksworld records sampled): natural-language symbolic planning. Output an action sequence like `(unstack red blue)`.
- **Graph Coloring** (200 problems generated by us, [src/oracle_graphcolor.py](src/oracle_graphcolor.py)): 3-coloring on 5–6 vertex graphs with varying edge densities (0.2–0.6). Assign each vertex one of {R, G, B} so adjacent vertices differ. Generated stratified across (n=5/6) × density.

(We previously also ran Countdown but the result there was uniformly 0% across conditions, dominated by exact-match scoring artifacts. We replaced it with Graph Coloring to get a more discriminating probe.)

### Eval conditions

For each dataset we run four conditions, all sharing the same Stage-2 LoRA checkpoint trained on G24-varied (`checkpoints/dagger_stage2_24_varied_bal_r4/z_s1234`):

| Condition | LoRA | z injection |
|---|---|---|
| **base** | off | none |
| **lora-no-z** | on | none |
| **lora-rand-z** | on | one Gaussian (norm = √hidden) at start |
| **lora-task-z** | on | per-task Stage-1 head's z, single-shot at start |

Per-task Stage-1 heads (one per task; LoRA stays the same): `checkpoints/head_{pronto,blocksworld,graphcolor}_qwen14b_rank/head.pt`. Built per the recipe in [docs/ood_datasets.md § 3](docs/ood_datasets.md). Driver: [src/eval_ood_generic.py](src/eval_ood_generic.py); scorer: [src/score_ood.py](src/score_ood.py); data prep: [data/prepare_ood_evals.py](data/prepare_ood_evals.py).

### Results

The full numbers (4-condition × 4-task plus PT-SFT, ToT, and Stage-1+2 in-domain) live in the **Headline results** section above. This section explains what the numbers mean.

**Z transfers in-domain, not OOD.** `LoRA-G24 (task-z) > LoRA-G24 (no-z)` only on G24-100 (+3 pp). On PQ/BW/GC, task-z ≤ no-z regardless of injection density. We tested this directly with dense per-step injection on BW + GC: the gap doesn't close (BW 33→36, GC 67→64; see [src/eval_dense_z.py](src/eval_dense_z.py)). Task-structure mismatch, not injection sparsity, is the bottleneck.

**The LoRA itself transfers on G24-similar tasks** even without z: GC +7 pp over base (constraint satisfaction with sequential decisions resembles G24's "pick op + check constraint" loop), PQ +3 pp (deductive logic), BW neutral.

**Random z is universally harmful.** On every task, `LoRA-G24 + rand-z` is the worst row. The z channel is *active* — uninformed perturbations matter — but uninformed-z degrades output (e.g., on PQ random-z makes the model abandon the "reply with one letter" rule and start free-form explaining; on BW random-z drops 7/100 plans to zero actions).

**Pipelines reproduced.** v1 of these experiments evaluated only `base` / `LoRA-no-z` / `LoRA-rand-z` (no task-specific Stage-1 heads); raw outputs in [results/eval_ood/](results/eval_ood/). v2 added per-task Stage-1 heads (`checkpoints/head_{pronto,blocksworld,graphcolor}_qwen14b_rank/head.pt`) so we have meaningful task-z; outputs in [results/eval_ood_v2/](results/eval_ood_v2/). Goal-reaching scoring (the meaningful BW metric — simulate the model's plan from initial state, check if `goal_facts ⊆ final_state`) is in [src/score_ood.py](src/score_ood.py); the strict exact-match metric drastically under-estimates BW correctness — see [docs/ood_datasets.md § 2 Scoring](docs/ood_datasets.md).

Followups still open:
- Train the LoRA on a *mixture* of arithmetic + deductive / planning tasks so it actually learns to read z across task types.
- Use a larger Stage-1 head (more capacity to distinguish OOD states).

Stage-1 head construction recipe + per-task tree caching commands: see [docs/ood_datasets.md § 3](docs/ood_datasets.md).

---

## Planning Tokens baseline ([Wang et al. 2023](https://arxiv.org/abs/2310.05707))

Method: prepend a discrete *planning token* before each reasoning step. Train SFT-style with the augmented data. Tokens summarize the step's "type" — for arithmetic problems the paper uses operator categories (`<PLAN:+>`, `<PLAN:->`, `<PLAN:*>`, `<PLAN:/>`) plus `<PLAN:ANS>`.

We reproduce + extend in two phases:

### Phase A — implementation correctness check (Phi-1.5 + GSM8K)

The paper reports Phi-1.5 + GSM8K: **baseline 12.5% → arithmetic-PT 15.0%** (+2.5pp).

Our run (Phi-1.5, LoRA r=16, 10 epochs, bf16, DDP-3 gloo on 3 GPUs ≈ 18 min/run; eval on 355/1319 records, see [src/train_sft_gsm8k.py](src/train_sft_gsm8k.py) + [src/eval_gsm8k.py](src/eval_gsm8k.py)):

| | Baseline (no PT) | Arithmetic PT | Δ |
|---|---|---|---|
| Phi-1.5 GSM8K (paper) | 12.5% | 15.0% | +2.5pp |
| Phi-1.5 GSM8K (ours) | **30.4%** | **32.1%** | **+1.7pp** |

Trend matches (small positive Δ for PT) — implementation verified. Absolute numbers higher because our prompt format / decoding setup differs (LoRA only, raw `Question:/Answer:` template, greedy).

### Phase B — Planning Tokens applied to the 3 OOD tasks

We adapt the arithmetic variant (5-token vocab) per task and SFT a Qwen2.5-14B + LoRA on each task's training data. Data prep: [data/prepare_pt_ood_data.py](data/prepare_pt_ood_data.py).

| Task | Planning tokens | Train data | Train size |
|---|---|---|---|
| **CD** | `<PLAN:+/-/*//>` per step + `<PLAN:ANS>` | `data/cd_train_sft.jsonl` (existing CD trajectories) | 1000 |
| **BW** | `<PLAN:PICKUP/PUTDOWN/STACK/UNSTACK>` + `<PLAN:ANS>` | PlanBench gold plans (records 200-449) | 250 |
| **PQ** | `<PLAN:DERIVE_TRUE/DERIVE_FALSE>` + `<PLAN:ANS>` | Oracle-generated proofs from forward chaining | 250 |

SFT: Qwen2.5-14B + LoRA r=16, DDP-2 gloo (each task on 2 GPUs in parallel) — total ~7 min wall for all 3.

#### Results

Numbers are in the **Headline results** section above (PT-SFT row). Three distinct failure modes:

- **PQ — format learning, not reasoning**: PT-SFT *underperforms* base (52.5 vs 60). Generations have correct planning-token format but mostly default to "Answer: A" regardless of input. The model learned to mimic the trajectory shape, not the deduction.
- **BW — surface-pattern fit, not planning**: PT-SFT 94.5% is not evidence of compositional planning. PlanBench BW has (a) only 4 action types, (b) short repetitive gold plans, (c) lenient goal-reaching scoring. 250 SFT examples are enough to fit the surface pattern. PQ + GC under the same recipe don't carry, because they need real deduction / combinatorial search.
- **CD — hallucinated arithmetic**: lenient string-match `Answer: 647` gives 58%; strict validation (math correct AND uses only pool numbers, via [src/evaluate_generic.py](src/evaluate_generic.py)) gives 0%. Model writes plausible arithmetic chains but injects numbers not in the pool ("562 + 85 = 647" when 85 isn't there).

**Bottom line:** PT-SFT does not give a clean "this is what good reasoning transfer looks like" baseline. BW wins but memorizes; CD is a scoring artifact; PQ regresses. Eval scripts: [src/eval_pt_ood.py](src/eval_pt_ood.py); raw outputs in [results/eval_pt_ood/](results/eval_pt_ood/).

---

## Pipeline end-to-end

### 0. Setup

```bash
pip install -r requirements.txt
```

**Hardware**: 8× NVIDIA A6000 (48 GB). GPUs 5↔7 have a broken NCCL pair on this node; training scripts default to `MEM_THRESHOLD=30000`-MiB auto-detect which usually picks a safe trio.

### 1. Data preparation — tree cache

Enumerates the full state tree for every solvable 24-problem in `data/24_{train,val,test}.jsonl` and caches (a) the tree metadata (`parents`, `depths`) and (b) the frozen base LLM's last-token hidden state for every node as float16 `.npy` memmaps.

```bash
BASE_MODEL=Qwen/Qwen2.5-14B-Instruct \
OUT_DIR=data/trees_qwen14b \
LOG_DIR=logs/gen_tree_qwen14b \
BATCH_SIZE=32 \
bash scripts/run_gen_tree_data.sh
```

- Sharded across all detected free GPUs (one python process per GPU), each shard handles `idx % world == rank` problems.
- Resume-safe: existing `data/trees_qwen14b/{split}/problem_{idx}.pt` + `hidden_{idx}.npy` files are skipped.
- Produces ~33 GB: `data/trees_qwen14b/{train,val,test}/` — 1090 train / 136 val / 136 test trees, ~3000 nodes per tree on average.

### 2. Stage 1 — head training + evaluation

```bash
python -m src.train_head --config configs/head_qwen14b_origin_ranking.yaml
python -m src.eval_head  --config configs/head_qwen14b_origin_ranking.yaml
```

Trains for 20 epochs on the cached hidden states (no LLM loaded), saves `checkpoints/head_qwen14b_origin_ranking/head.pt`, then produces:

- `results/head_eval/qwen14b_origin_ranking/metrics.json` — Spearman rank correlation of `|z|` vs `v(s)`, origin-distance histograms (val + test).
- `results/head_eval/qwen14b_origin_ranking/vis_tree_{idx}.png` — 2D tangent-PCA visualization of example trees.

### 3. Stage 2 — DAgger training + inference + eval

```bash
# Per-run per-seed launcher. Requires a stage-1 head first.
BASE_CONFIG=configs/stage2_dagger_qwen14b_stable.yaml \
bash scripts/run_train_stage2_dagger.sh <noz|z> qwen14b_origin_ranking [seed]
```

Each invocation trains one run (z-injected or no-z control) for one seed. The driver (a) trains the LoRA + UpProjector across all detected free GPUs with DDP (manual gradient averaging — see *Training notes* below), (b) generates 100 test-problem solutions with `src.generate_24_stage2`, (c) validates them with `src.evaluate_24`.

Artifacts:
- `checkpoints/dagger_stage2_{head_tag}/{noz|z}_s{seed}/` — LoRA + UpProjector.
- `results/dagger_stage2_{head_tag}/{noz|z}_s{seed}/{generations.jsonl, metrics.json, rollout_stats_epoch*.json}`.

Inference-time `--random_z` is supported via `src.generate_24_stage2` for sanity-checking a trained z-run checkpoint.

### Evaluation

`src.evaluate_24` validates each generated 3-step solution: parses `Step N: a op b = r`, replays the arithmetic, checks that all 4 input numbers are used exactly once, and confirms the final result is 24. Accuracy = fraction of problems with a valid 3-step solution. All runs default to 100 held-out test problems from `data/24_test_tot.jsonl`. Scale via `--limit`.

---

## Headline results

All numbers below: Qwen-2.5-14B-Instruct unless noted, 100 held-out problems per task, single seed (1234), greedy decoding for all non-search rows.

### Game-24 in-domain (varied-target task is what the LoRA is trained on)

| System | Accuracy | Inference compute / problem |
|---|---|---|
| Qwen-2.5-14B fewshot (no LoRA) | 0.11 | 1 greedy decode |
| Qwen3-14B fewshot (non-thinking) | 0.05 | 1 greedy decode |
| Qwen-2.5-14B SFT (token-matched) | 0.16 | 1 greedy decode |
| Qwen-2.5-14B ToT any-of-5 | 0.16 | 5 × (propose + 3 × value) |
| Qwen-2.5-14B ToT top-1 | 0.01 | same |
| **Qwen3-14B ToT any-of-5** | **0.60** | 5 × (propose + 3 × value) |
| Qwen3-14B ToT top-1 | 0.49 | same |
| **Qwen-2.5-14B HypPlan DAgger noz-stable** | **0.57** | **1 greedy decode** |
| Qwen-2.5-14B HypPlan DAgger z-stable | 0.55 | 1 greedy decode |
| Qwen-2.5-14B HypPlan + LoRA-G24 (task-z) on G24-100 | 0.12 | 1 greedy decode |

At 14B scale, HypPlan's DAgger-trained LoRA (top-1 greedy) effectively ties Qwen3's ToT any-of-5 (0.60) and beats Qwen3's ToT top-1 (0.49) with ~15× less inference compute. GPT-4 sits ~14–17 pp above our best 14B number.

### Cross-task (G24-trained LoRA + per-task heads, plus in-domain baselines)

This is the "does the methodology transfer / does it work in-domain on other tasks" picture. PT-SFT and HypPlan-indom train per-task; LoRA-G24 rows reuse the *same* G24-trained LoRA across all OOD tasks.

| Condition | G24-100 | PQ-100 | BW (goal) | GC-100 |
|---|---|---|---|---|
| base (no LoRA, no z) | 11 | 60 | 41 | 61 |
| LoRA-G24 (no z) | 9 | 63 | 43 | 68 |
| LoRA-G24 (rand z) | 4 | 43 | 35 | 60 |
| LoRA-G24 (task z, single-shot) | **12** | 62 | 33 | 67 |
| LoRA-G24 (task z, dense per-step) | 12 | n/a | 36 | 64 |
| PT-SFT in-domain (per-task) | 6 | 52.5 | **94.5*** | 64 |
| ToT (top-1) | 1 | 41 | 58 | 34 |
| ToT (any-of-5) | 16 | 44 | 83 | 56 |
| **HypPlan Stage-1+2 in-domain (per-task)** | – | **75** | 10 | **88** |

*PT-SFT BW=94.5% is memorization on PlanBench gold (same distribution as test). Not evidence of compositional planning. See [docs/ood_datasets.md §5](docs/ood_datasets.md#5-planning-tokens-pt-sft-baseline).

Reading the table:
- **Z transfers in-domain, not OOD**: `LoRA-G24 (task-z) > LoRA-G24 (no z)` only on G24 (+3 pp). On PQ/BW/GC, task-z ≤ no-z.
- **The G24-trained LoRA itself transfers** to G24-similar tasks even without z: GC +7 pp, PQ +3 pp, BW neutral.
- **Random z is universally harmful** — z channel is "active" but uninformed-z degrades output.
- **Stage-1+2 in-domain training is a strong baseline on PQ/GC** (+15 / +27 pp over base). BW is the open problem; failure analysis in [docs/ood_datasets.md §6](docs/ood_datasets.md#6-hypplan-stage-12-trained-in-domain-per-task).

Ablation details and the stability story behind the `noz-stable` / `z-stable` numbers are in *Experiments* below.

---

## Current focus: Qwen-2.5-14B-Instruct as the shared base

Base model chosen after running few-shot validation on several candidates:

| Model (greedy + 3-shot Game-24) | Accuracy | Format rate |
|---|---|---|
| Llama-3.1-8B-Instruct | 3% | 96% |
| **Qwen-2.5-14B-Instruct** | **11%** | **100%** |

Qwen-2.5-14B-Instruct gives strong few-shot accuracy at zero SFT cost and a clean "same base as ToT" story. The pivot: both HypPlan and the ToT baseline use `Qwen/Qwen2.5-14B-Instruct` as the frozen base. This lets us skip the SFT preprocessing step and compare like-for-like against ToT.

Few-shot validator: [scripts/fewshot_baseline.py](scripts/fewshot_baseline.py) + [scripts/run_fewshot_baseline.sh](scripts/run_fewshot_baseline.sh).

Code plumbing that threads the current base + prompt through the pipeline:
- [src/prompt_builders.py](src/prompt_builders.py) — `sft_prompt_24`, `fewshot_chat_prompt_24`, `sft_prompt_cd`, `fewshot_chat_prompt_cd`. Each builder returns `(text, add_special_tokens)` so chat-template prompts don't double-add BOS.
- [src/train_head.py](src/train_head.py) supports `origin_ranking_rank` loss (scale-invariant rank-based margin for wide-range Countdown v).
- [src/dagger_rollout.py](src/dagger_rollout.py) / [src/train_stage2_dagger.py](src/train_stage2_dagger.py) / [src/generate_24_stage2.py](src/generate_24_stage2.py) thread `prompt_builder` through rollout, training and inference. Config key: `training.prompt_style` (`"sft"` or `"fewshot"`).

---

## Experiments

All experiments below use `Qwen/Qwen2.5-14B-Instruct` as the frozen base model unless stated otherwise, 100 held-out Game-24 problems, seed 1234 (n=1 unless stated), 2-GPU DDP.

### 1. HypPlan (stage 1 + stage 2) on Game-24

- Stage-1 head: `configs/head_qwen14b_origin_ranking.yaml`.
- Stage-2 DAgger: `configs/stage2_dagger_qwen14b.yaml` (unstable), `configs/stage2_dagger_qwen14b_stable.yaml` (final, lr=5e-5, clip=0.3).
- Artifacts: `checkpoints/head_qwen14b_origin_ranking/`, `checkpoints/dagger_stage2_qwen14b_origin_ranking/{z,noz}_s1234/`, `results/dagger_stage2_qwen14b_stable/{z,noz}_s1234/`.

**Final numbers (stable hyperparams, lr=5e-5, clip=0.3):**

| Run | Accuracy |
|---|---|
| noz-stable | **0.57** |
| z-stable | 0.55 |

The original lr=1e-4, clip=1.0 run gave z=0.43, noz=0.00 (catastrophic collapse). Investigation below.

#### 1a. The noz collapse and the stability fix

The original run at lr=1e-4, clip=1.0 showed a dramatic z / noz gap (0.43 vs 0.00). Post-hoc analysis:

- **Loss spike at epoch-2 step 6600.** `train.jsonl` shows the 50-step average loss jumped from `0.0371` to `3.5657` and never recovered — plateaued at ~0.75–0.95 for the remaining ~1000 pairs. The z run's loss stays ≤ 0.052 throughout epoch 2.
- **Damage localised to layers 0–1.** Comparing the final LoRA adapter weights between z and noz, `max_abs` per tensor is ~2× higher in noz for exactly 6/384 tensors, all q/k/v/o_proj in transformer layers 0–2. Overall L2 norms are nearly identical (50.64 vs 50.91), so the damage is concentrated rather than diffuse.
- **Rollout stats indistinguishable.** `rollout_stats_epoch*.json` is nearly identical between z and noz: same n_rollouts, similar empty-oracle rate (~30–35%), low drop rates (<2.5%). Harm shows up only in the epoch-2 training phase.
- **grad_clip=1.0 was insufficient.** Adam updates with clipped-to-norm-1 gradients still moved a handful of weights in the input-side attention layers enough to break the model.

**Mechanism hypothesis.** z-injection provides an input-side anchor at each step boundary that stabilises layer-0 attention across the long chat-template + 3-shot + problem prompt (~700 tokens). Without that anchor, a hard training pair late in epoch 2 knocked layers 0–1 off-manifold and the model could not recover — hence "always emits Answer: 24" (late layers still try to close) with hallucinated intermediate arithmetic (early layers can no longer read the input pool).

**Track A (2026-04-22): pre-spike `lora_epoch1` snapshot gives 0.28** on the same 100 held-out problems — confirming the collapse hypothesis. noz was learning fine through epoch 1 (between SFT 0.16 and z-run 0.43); the epoch-2 training spike is what wiped it to 0.00. Artifacts: `results/dagger_stage2_qwen14b_origin_ranking_noz_ep1eval/metrics.json`.

**Track B (2026-04-22): stable retrain inverts the z-vs-noz story.** Rerunning at `lr=5e-5, grad_clip=0.3` kept both arms stable through all 3 epochs — no loss spike, no layer-0 damage — giving noz-stable 0.57 and z-stable 0.55.

**This rewrites the headline.** At n=1 seed, apples-to-apples, `noz` *beats* `z` by 2 pp. What made the original lr=1e-4 z-run hit 0.43 wasn't the geometric signal from stage 1 — it was the z-injection incidentally stabilising an otherwise unstable optimisation, at the cost of peaking below the stable-training ceiling. At the correct LR, the **DAgger loop alone** (no z, no head, no `UpProjector`) gives the best Game-24 accuracy we've measured on Qwen-14B. Caveat: n=1 seed; 2 pp is within plausible seed noise. A 3-seed rerun of `{noz-stable, z-stable}` is the next cheap experiment to make this crisp.

### 2. ToT baselines at 14B scale

#### 2a. Qwen-2.5-14B ToT (matches HypPlan's base)

- Launcher: `bash scripts/run_tot_qwen14b.sh` — single-model ToT (`--shared_model --use_chat_template`, paper-default `n_generate=1, n_evaluate=3, n_select=5, T=0.7`).
- Reports both any-of-top-5 (matches ToT paper's 74% metric) and top-1. Overlap with `data/24_test_tot.jsonl` is 100/100.
- Artifacts: `results/tot_baseline_qwen14b/seed_1234/`.

| Metric | Accuracy |
|---|---|
| any-of-top-5 | 0.16 |
| top-1 | 0.01 |

Much weaker than the ToT paper's 74% on GPT-4 — consistent with prior findings that open 14B models are a lot less capable than GPT-4 on this benchmark.

#### 2b. Qwen3-14B ToT (stronger search baseline, same model class)

Probed Qwen3-14B (newer generation with hybrid thinking mode) as a base on 2026-04-22.

| Task | Qwen2.5-14B | Qwen3-14B (non-thinking) | Δ |
|---|---|---|---|
| Game-24 fewshot | 0.11 | 0.05 | −6 pp |
| Countdown fewshot | 0.00 | 0.00 | — |
| **Game-24 ToT any-of-top-5** | **0.16** | **0.60** | **+44 pp** |
| **Game-24 ToT top-1** | **0.01** | **0.49** | **+48 pp** |

`enable_thinking=False` is passed through `apply_chat_template` (via a small `TypeError`-guarded wrapper in `src/prompt_builders.py::_apply_chat_template_no_think`, mirrored in `scripts/fewshot_baseline.py` and `src/tot_baseline.py`), so the model never emits `<think>...</think>` blocks — required because our parser and DAgger oracle consume step-by-step output. Also fixed a crash in `src/tot_baseline.py::trajectory_to_generation` where Qwen3 occasionally emits truncated decimals like `12.333...` that broke `int(float(...))`.

**Why Qwen3 wins ToT but loses single-shot.** Qwen3's non-thinking-mode *evaluator* (the "sure / likely / impossible" value call that ToT uses to rank branches) is dramatically better than Qwen2.5's — it actually picks correct trajectories top-1 49/100 times vs Qwen2.5's 1/100. The *generator* (single-shot greedy) is slightly worse because Qwen3's direct-answer training gets locked behind the thinking channel that we're disabling. Qwen3 at 0.60 any-of-5 is the closest open model we have to the ToT paper's 0.74 GPT-4 number — the gap is now 14 pp, not 58 pp.

**Decision — keep Qwen2.5-14B for the DAgger pipeline.** DAgger uses the base model as a *generator* during rollout, not as an evaluator — it can't exploit Qwen3's ToT strength. With Qwen3's generator at only 0.05 fewshot, expected DAgger ceiling is much lower than Qwen2.5's 0.57 headline. Qwen3 non-thinking ToT joins the baseline table as a stronger search contender at the same model class.

### 3. Token-matched SFT baseline (Qwen-14B)

Purpose: a fair same-data SFT baseline for HypPlan. The canonical training corpus is `data/24_train.jsonl` (1090 problems × 1 canonical trajectory). With 3 epochs this yields ~3k sequence updates and ~200k supervised tokens, comparable to DAgger's ~300k supervised tokens (15k pairs × 20 tokens).

- Base: Qwen-2.5-14B-Instruct (frozen) + fresh LoRA (`r=16, α=32, dropout=0.05`, targets `q/k/v/o_proj`) — identical to DAgger LoRA.
- Prompt: `fewshot_chat_prompt_24` at both training and eval — identical to DAgger.
- Optimizer: AdamW lr=1e-4, 3 epochs, cosine schedule, bf16, grad_clip=1.0, effective batch 8 (bs 1 × 2 GPUs × grad_accum 4).
- Loss: next-token CE supervising only the assistant completion (prompt tokens labelled `-100`).
- Inference: `src/generate_sft_qwen.py` — loads LoRA, same `fewshot_chat_prompt_24`, greedy decode, `parse_and_validate` from [src/evaluate_24.py](src/evaluate_24.py).
- Code: `src/dataset_24.py::Game24SFTChatDataset`, [src/train_sft_24_qwen.py](src/train_sft_24_qwen.py), [scripts/run_sft_qwen14b.sh](scripts/run_sft_qwen14b.sh), [configs/sft_24_qwen14b.yaml](configs/sft_24_qwen14b.yaml).
- Artifacts: `checkpoints/sft_24_qwen14b/`, `results/sft_24_qwen14b/`.

**What's held fixed vs DAgger:** base model, LoRA config, prompt builder, optimiser, DDP setup, test set, parser, supervised-token budget (≈200k SFT vs ≈300k DAgger). **The only thing that differs is the training signal:** SFT supervises next-token CE on one canonical oracle trajectory per problem; DAgger rolls out under the current policy, queries the oracle at each reached state for *all* winning ops, trains CE on the picked winner, and injects the stage-1 `z` at each step boundary.

**Result (seed 1234, 2-GPU DDP, 3 epochs, 408 optim steps, ~12 min):** Qwen-14B SFT = **0.16**, vs Qwen-14B DAgger z-run = **0.43** (+27 pp at identical LoRA / optimiser / prompt). The gain isn't coming from the adapter or extra supervised tokens — it's the DAgger rollout-and-relabel loop.

### 4. Geometry ablation (Poincaré vs Euclidean vs random z)

Purpose: isolate whether hyperbolic geometry is doing real work on Qwen-14B. Swap the manifold to flat Euclidean keeping every other knob (stage-1 loss, stage-2 DAgger schedule, head architecture, base model) fixed.

- Euclidean stage-1 head: `configs/head_qwen14b_euclidean_origin_ranking.yaml` — `manifold: euclidean`, same `loss: origin_ranking, hyp_dim: 32, head_hidden_dims: [1024, 256], lr: 1e-3, epochs: 20`. Trains on the same cached Qwen-14B hidden states in `data/trees_qwen14b`.
- Euclidean stage-2 DAgger: `configs/stage2_dagger_qwen14b_euclidean.yaml`. Z-run only (noz is manifold-agnostic).
- Random-z: `--random_z` added to `src/train_stage2_dagger.py`, propagates through `src/dagger_rollout.py`, persisted in saved config so `src/generate_24_stage2.py` picks it up at eval. Norm-matches the trained `z_inj` output of `UpProjector` (L2 ≈ √hidden_dim ≈ 72 after final LayerNorm).

**Summary (seed 1234, identical Qwen-14B base + LoRA + prompt + optimiser + DAgger schedule; only the injection at step boundaries differs):**

| Injection | Accuracy | What it tells us |
|---|---|---|
| none (`noz`, original lr=1e-4) | 0.00 | Catastrophic LoRA collapse — pipeline needs *something* at each step boundary |
| norm-matched random Gaussian (`randz`) | 0.42 | Unstructured anchor is enough to recover 42/100 |
| Euclidean `z` from stage-1 head | 0.46 | Flat-geometry structured z |
| Poincaré `z` from stage-1 head | 0.43 | Hyperbolic structured z |

**Inference-time random-z on the trained z-run LoRA.** Reusing the trained z-run LoRA checkpoint but replacing the injected `z` with a fresh unit-norm random vector at every step boundary gives **0.44** — statistically tied with the real trained `z` (0.43). So on Qwen-14B + few-shot-chat, the specific *direction* of the geometric `z` is not what's doing the work at inference time; any vector of similar norm suffices.

**Interpretation.** At n=1 seed on Qwen-14B + few-shot chat, the geometric content of `z` is empirically irrelevant: random, Euclidean and hyperbolic all land within ±3 pp. The ~42 pp jump from `noz` (0.00) to `randz` (0.42) is not a signal gain — it's an anchor gain. The specific mechanism on this base model appears to be: injecting *any* non-zero virtual token at every step boundary keeps layers 0–1 out of the degenerate basin that `noz` epoch-2 training falls into, and the LoRA + DAgger loop do the rest.

This does not contradict the earlier Llama-3.1-8B result (Poincaré +7.7 ± 4.2 pp over no-z) — there the no-z run was stable (0.333) and the measured effect was a structural bonus on top of a stable baseline. On Qwen-14B we can't repeat that test cleanly until the `noz` instability is closed (the stable-retrain Track B is the closest *stable* counterfactual we have). Plan: rerun `noz-stable`, `randz`, `z_Hyp`, `z_Euc` at 3 seeds — the minimum needed to credibly separate "stabiliser" from "signal" on Qwen.

Artifacts: `checkpoints/head_qwen14b_euclidean_origin_ranking/`, `checkpoints/dagger_stage2_qwen14b_euclidean/z_s1234/`, `checkpoints/dagger_stage2_qwen14b_origin_ranking/randz_s1234/`, `results/dagger_stage2_qwen14b_{euclidean,origin_ranking}/…`.

### 5. Task-generalisation probe: Game-24 stage-2 on Countdown (zero-shot)

Motivation: does the Game-24-trained stage-2 LoRA transfer to Countdown (6-number pool, variable integer target, 5 steps)? Zero-shot — no CD-specific training, just swap the test set and the fewshot examples in the prompt. All three runs use Qwen-2.5-14B-Instruct + `fewshot_chat_prompt_cd` on `data/cd_test.jsonl` (100 held-out problems).

| Run | LoRA | z-injection | Accuracy |
|---|---|---|---|
| Exp 1: base fewshot | — | — | **0.00** |
| Exp 2: Game-24 LoRA, no z | Game-24 DAgger z-run | off | **0.00** |
| Exp 3: full Game-24 stack on CD | Game-24 DAgger z-run | Game-24 head + up_proj, z at 5 CD boundaries | **0.00** |

All three hit 0 — but the three runs fail in *different* ways:

| Diagnostic (out of 100 problems) | Exp 2 (no z) | Exp 3 (with z) |
|---|---|---|
| Emits exactly 5 `Step N:` lines | 93 | 99 |
| All 5 arithmetic equations correct | 67 | 70 |
| All operands came from the pool | 83 | 87 |
| Final step's arithmetic actually equals target | **3** | **4** |
| Final step's `= r_target` value *claims* to equal target | 19 | 26 |
| *Cheats*: claims target but arith is wrong | **16** | **22** |

**What transferred.** Output format, step counting, correct integer arithmetic, pool-only operands — all 67–99%. The LoRA learned a domain-general "emit arithmetic step chains" skill.

**What did not transfer.** *Target-conditioned planning.* Only 3–4% of trajectories land on the target. Game-24's target is always 24, so the LoRA never needed to read the target token during stage 2 — it learned "always end with `Answer: 24`". On Countdown, the model plows through five reasonable steps regardless of target.

**The most telling failure: cheating.** ~20% of generations end with a step whose *printed* right-hand side equals the problem's target, but whose *actual* arithmetic does not. Concrete example (Pool `[1, 6, 7, 9, 9, 25]`, Target 554):

```
Step 1: 6 * 7 = 42. Remaining: 1 9 9 25 42
Step 2: 9 + 9 = 18. Remaining: 1 18 25 42
Step 3: 1 + 18 = 19. Remaining: 19 25 42
Step 4: 19 + 25 = 44. Remaining: 42 44
Step 5: 42 + 44 = 554. Answer: 554
```

Steps 1–4 arithmetically correct; step 5 is `42 + 44 = 86`, not 554 — the LoRA forces the last line to end in `554` because that's the string it was told to hit. Another, target=247: `Step 5: 150 + 39 = 189. Answer: 247` — final arith equals 189 but the model appends `Answer: 247` anyway. Both behaviours are symptoms of the same fact: the LoRA treats the target as a string it must emit, not as a planning constraint.

**Effect of z-injection (Exp 3 vs Exp 2).** Using the Game-24 head + up_projector with CD state text nudges format/arithmetic metrics up by 3–7 pp, but the planning metric (real final = target) barely moves (3 → 4). Consistent with the Qwen Game-24 finding that z is primarily a stabiliser, not a planner — it doesn't add target-conditioning the LoRA lacks.

**Implications for the full CD training run.** The CD DAgger pipeline is the right next step, because the stage-2 training signal *does* depend on the target (the oracle's winning ops are a function of `target`). A CD-native LoRA will be forced to learn target-conditioned planning by construction, which is exactly the skill Game-24's LoRA is missing. Artifacts: `results/fewshot_qwen14b_cd/`, `results/transfer_24lora_cd_noz/`, `results/transfer_24lora_cd_z/`.

---

## Why DAgger (why not teacher forcing?)

Our initial Stage-2 design trained the LoRA on **teacher-forced** trajectories — injecting `z` at step boundaries and optimizing standard next-token CE. That run (preserved under `results/hyp_stage2_*`) landed at 0.21 accuracy — statistically indistinguishable from a null baseline (random `z`). Two compounding reasons:

1. **Teacher forcing eliminates the uncertainty z was designed for.** At each step boundary during training, the model is conditioned on the *correct* preceding trajectory. z — a compressed summary of that same trajectory — is informationally redundant given the context the LLM already has. CE has no gradient pressure to extract z's content.
2. **CE does not reward z-usage.** The LM can drive CE low via alternative paths (preceding text, base priors). Nothing forces the policy to *depend* on z.

Evidence from null-baseline experiments: LoRA trained with real z broke when given random z at test (the "+9 pp" figure we initially misinterpreted), and LoRA trained with random z worked normally with random z. That pattern means the LoRA learned z's **distributional statistics** (norm, variance), not its **semantic content** — it used z as a calibration signal, not a payload.

**DAgger fixes this** by treating the head as a privileged critic (Stage 1 had access to the enumerated tree — solution locations, distance to nearest success) and the LoRA as a policy trained under its own state distribution. Under free generation the model reaches genuinely uncertain states; z then carries decision-relevant information the model cannot trivially recompute from context.

### Two-run experimental design

Both runs use **identical** code path, warm start, sampling hyperparams, oracle rules, and DAgger schedule. A single `--use_z` flag toggles z-injection on/off. This isolates z's contribution. The clean metric: `Δ_accuracy = acc(z run) − acc(no-z run)`.

### Warm start

- Frozen base — gives 0.11 (Qwen-14B fewshot) or 0.12 (legacy SFT-merged Llama) at step 0.
- **Fresh LoRA** with standard PEFT init (A ∼ 𝒩, B = 0, so delta = 0 at step 0).
- **Small-std-init UpProjector** (σ=1e-3 on final Linear's weight, bias=0). Initially tried fully zero-init but `LayerNorm(0)` combined with the Instruct chat template triggered a degenerate fallback where the model emitted `"assistant\n..."` at step 1. A tiny non-zero init sidesteps this while staying close to "no effect" at step 0.
- Frozen `head_{manifold}_origin_ranking` as the critic.

First rollout with this init ≈ pure frozen-base behavior, without inheriting any bad z-attention habits. DAgger teaches the LoRA to use z from scratch.

### Decisions locked in

1. Drop invalid trajectories from the invalid step onward. Log drop rate per epoch; alarm if >50% after epoch 0.
2. Single-winner CE (phase 1, lex tiebreak).
3. Fresh LoRA (B=0) + small-std-init UpProjector (σ=1e-3).
4. T=0.7, top-p=0.95 for rollout. Greedy for eval.
5. Lockstep DAgger: per epoch, rollout all 1090 train problems (3 trajectories each ≈ 3300 trajectories), then one CE pass over collected pairs. Repeat for 3 epochs. Both runs execute in parallel.

### Components

- [src/oracle_24.py](src/oracle_24.py) — given `remaining`, returns winning next-ops via memoized recursive search (not a tree-file lookup). Handles any state the model can reach, including off-tree sequences.
- [src/dagger_rollout.py](src/dagger_rollout.py) — one-problem rollout: token-by-token sampling with per-step z injection, step parsing, oracle labeling, invalid-step detection, tolerant regex for z-injection prefix artifacts.
- [src/train_stage2_dagger.py](src/train_stage2_dagger.py) — two-run Stage-2 (DAgger) trainer. `--use_z`, `--seed`, `--random_z`. Manual gradient averaging under DDP. NCCL collective timeout raised to 60 min to tolerate rank imbalance during variable-length rollouts.
- `configs/stage2_dagger_qwen14b_stable.yaml` — current Stage-2 config template.
- [scripts/run_train_stage2_dagger.sh](scripts/run_train_stage2_dagger.sh) — per-run per-seed launcher: `bash run_train_stage2_dagger.sh <noz|z> <head_tag> [seed]`.

See [docs/dagger_walkthrough.md](docs/dagger_walkthrough.md) for a concrete example walkthrough (rollout terminology, oracle mechanics, z vs no-z run side-by-side on problem `4,5,6,10`).

### Distributed training notes

Stage-2 DDP uses **manual gradient averaging** rather than `torch.nn.parallel.DistributedDataParallel`:

- Seed `torch.manual_seed(1234)` before LoRA + `UpProjector` init so every rank gets identical weights without a broadcast collective.
- After `loss.backward()`, iterate over trainable params and call `dist.all_reduce(p.grad, op=SUM); p.grad.div_(world_size)` before `optimizer.step()`.

Why not standard DDP? Stage-2's computation graph changes per iteration (variable-K per-boundary inner loop, plus `disable_adapter()` sub-forwards for state encoding). That makes DDP's bucket-ready ordering diverge across ranks and deadlock the first auto-reduce. Manual averaging sidesteps this; the sync cost is trivial for our ~22M trainable params.

NCCL topology gotcha on this host: GPUs 5↔7 are a broken pair at the NCCL level (works pair-wise with other GPUs; deadlocks when both are in the same process group). If you must use all 8 GPUs, verify with a `scripts/test_nccl.sh`-style probe first.

---

## Project layout

```
HypPlan/
├── configs/
│   ├── head_qwen14b_origin_ranking.yaml          # stage-1 (Poincaré) on Qwen-14B
│   ├── head_qwen14b_euclidean_origin_ranking.yaml # stage-1 (Euclidean) ablation
│   ├── stage2_dagger_qwen14b_stable.yaml         # stage-2 DAgger, stable hyperparams
│   └── ...                                        # earlier configs retained for reproducibility
├── src/
│   ├── tree_data.py           # enumerate_tree, render_state, pair_distances_lca
│   ├── hyperbolic.py          # Lorentz ops
│   ├── head.py                # HyperbolicHead (Poincaré/Lorentz/Euclidean) + UpProjector
│   ├── train_head.py          # stage-1 trainer (origin_ranking / origin_ranking_rank)
│   ├── eval_head.py           # Spearman(|z|, v(s)) + 2D viz
│   ├── prompt_builders.py     # sft_* and fewshot_chat_* prompt builders
│   ├── oracle_24.py           # stage-2 oracle: winning_ops(remaining)
│   ├── dagger_rollout.py      # stage-2 rollout + oracle labeling (threaded prompt_builder)
│   ├── dataset_24_stage2.py   # per-boundary canonical state tokenization
│   ├── train_stage2_dagger.py # stage-2 (DAgger) trainer (two-run, DDP, --random_z)
│   ├── generate_24_stage2.py  # inference (supports --no_z_inject, --random_z)
│   ├── generate_sft_qwen.py   # SFT-baseline inference (Qwen chat prompt)
│   ├── tot_baseline.py        # ToT BFS runner (paper-faithful, chat-template wrap)
│   └── evaluate_24.py         # solution validator
├── data/
│   ├── generate_tree_data.py  # offline tree + hidden-state cache builder
│   ├── 24_{train,val,test}.jsonl
│   └── trees_qwen14b/         # cached tree metadata + Qwen-14B hidden states
├── docs/
│   └── dagger_walkthrough.md  # concrete example of stage-2 mechanics
├── scripts/
│   ├── run_gen_tree_data.sh
│   ├── run_train_head.sh
│   ├── run_train_stage2_dagger.sh
│   ├── run_tot_qwen14b.sh
│   └── run_sft_qwen14b.sh
├── checkpoints/
│   ├── head_qwen14b_*/        # stage-1 heads (Poincaré / Euclidean)
│   ├── dagger_stage2_qwen14b_*/{z,noz}_s{seed}/
│   └── sft_24_qwen14b/        # token-matched SFT baseline
└── results/
    ├── head_eval/qwen14b_*/
    ├── dagger_stage2_qwen14b_*/
    ├── tot_baseline_qwen14b/
    ├── tot_baseline_qwen3_14b/
    └── sft_24_qwen14b/
```

<!-- Old v1 files (`train_plan_24.py`, `generate_24_plan.py`, `train_stage1.py`, `train_stage2.py` — the teacher-forced precursor, …) remain in place as reference — not deleted so prior `results/` stay reproducible. -->

---

## Open research questions / backlog

- **Phase-2 loss upgrade.** Replace single-winner CE with log-of-sum over full step texts of all winners (≈K× compute). Not needed for current numbers but cleaner theoretically.
