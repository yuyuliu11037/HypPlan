# HypPlan: Tree-Distortion Hyperbolic Planning for LLM Reasoning

A two-stage framework. **Stage 1** teaches a small head to embed reasoning-tree states into a hyperbolic space so that `|z|` tracks solution-proximity — pure geometric supervision on an enumerated state tree, no language-model loss. **Stage 2** trains a fresh LoRA + `UpProjector` on top of a frozen base LLM using **DAgger with a tree oracle**: the current policy rolls out trajectories freely, the oracle labels winning ops at each reached state, and CE trains the LoRA on those labels. The frozen head's geometric `z` is injected as a virtual token before each step boundary.

Runs on **Game of 24**. The MATH pipeline from the original HypPlan also lives in this tree (see `src/train_stage1.py` etc.) but the active pipeline is the two-stage Game-of-24 flow documented below.

Active base model: **Qwen-2.5-14B-Instruct** (hidden_dim=5120, frozen, used with a 3-shot chat prompt). Earlier Llama-3.1-8B-based runs are archived at the bottom of this README for reference.

---

## The two stages

### Stage 1 — hyperbolic head (LLM frozen)

For each Game-of-24 problem we enumerate the full state tree (root = initial 4 numbers, children = all legal (a, op, b) applications, leaves = 1-number terminal states). Each node's state text is encoded by the frozen base LLM. A small head MLP maps that hidden vector to a low-dim point (default `hyp_dim=32`) in a hyperbolic space via exp-map at the origin.

**Loss: `origin_ranking`.** A margin hinge on distance-to-origin, with target `v(s)` = BFS edge distance from state `s` to the nearest success leaf in the enumerated tree. For any sampled pair `(s_i, s_j)` with `v(s_i) < v(s_j)`:

`L = max(0, d_H(z_i, 0) − d_H(z_j, 0) + margin)`

This makes `|z|` track solution-proximity: states closer to a solution are pulled toward the origin, states farther are pushed outward. Supported manifolds: Poincaré ball and Lorentz hyperboloid.

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

The base pipeline above trains the LoRA on one fixed task: four numbers, target 24. That's what we measured (0.57 DAgger noz, 0.55 z). But the LoRA might have memorized "always reach 24" rather than learned a general skill — *use the z signal to steer reasoning*.

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

---

## OOD generalization probes: ProntoQA + Blocksworld + Graph Coloring

To probe whether the G24-trained LoRA transfers to OOD tasks, we run three test sets covering radically different reasoning types:

- **ProntoQA** ([renma/ProntoQA](https://huggingface.co/datasets/renma/ProntoQA), 200 records sampled): synthetic deductive reasoning. Given facts and rules, decide whether a statement is True (A) or False (B).
- **Blocksworld** ([tasksource/planbench](https://huggingface.co/datasets/tasksource/planbench), `task_1_plan_generation`, 200 blocksworld records sampled): natural-language symbolic planning. Output an action sequence like `(unstack red blue)`.
- **Graph Coloring** (200 problems generated by us, [src/oracle_graphcolor.py](src/oracle_graphcolor.py)): 3-coloring on 5–6 vertex graphs with varying edge densities (0.2–0.6). Assign each vertex one of {R, G, B} so adjacent vertices differ. Generated stratified across (n=5/6) × density.

### Eval conditions

For each dataset we run three baseline conditions on the original test prompt, all sharing the same Stage-2 LoRA checkpoint trained on G24-varied (`checkpoints/dagger_stage2_24_varied_bal_r4/z_s1234`):

| Condition | LoRA | z injection |
|---|---|---|
| **base** | off | none |
| **lora-no-z** | on | none |
| **lora-rand-z** | on | one Gaussian (norm = √hidden) at start |

We do not report "task-z on the original test prompt": the original prompts (single-letter for PQ; PDDL action lines for BW; "V_i = color" lines for GC) don't share G24's `Step N: a op b = r. Remaining: …` boundary structure that the LoRA was trained to read z at. To probe whether a meaningful task-specific z helps with a *matched* prompt, we use a CoT-style prompt for PQ/BW/GC — see the CoT-prompt rows in the headline cross-task table.

Per-task Stage-1 heads (one per task; LoRA stays the same): `checkpoints/head_{pronto,blocksworld,graphcolor}_qwen14b_rank/head.pt`. Built per the recipe in [docs/ood_datasets.md § 3](docs/ood_datasets.md). Driver: [src/eval_ood_generic.py](src/eval_ood_generic.py); scorer: [src/score_ood.py](src/score_ood.py); data prep: [data/prepare_ood_evals.py](data/prepare_ood_evals.py).

### Results

The full numbers (cross-task plus PT-SFT, ToT, and Stage-1+2 in-domain) live in the **Headline results** section above. This section explains what the numbers mean.

**Dense per-step z does NOT help OOD even with a matched CoT prompt.** On PQ, the controlled comparison `LoRA-G24 + CoT prompt` is 74% no-z vs 67% with dense per-step task-z (−7 pp). The geometric z signal genuinely doesn't transfer.

**The LoRA itself transfers on G24-similar tasks** even without z: GC +7 pp over base (constraint satisfaction with sequential decisions resembles G24's "pick op + check constraint" loop), PQ +3 pp (deductive logic), BW neutral. With the CoT prompt, the no-z PQ number jumps to 74% — most of the in-domain gain comes from the prompt change, not from task-specific Stage-2 training.

**Random z is universally harmful.** On every task, `LoRA-G24 + rand-z` is the worst row. The z channel is *active* — uninformed perturbations matter — but uninformed-z degrades output (e.g., on PQ random-z makes the model abandon the "reply with one letter" rule and start free-form explaining; on BW random-z drops 7/100 plans to zero actions).

**Pipelines reproduced.** v1 of these experiments evaluated only `base` / `LoRA-no-z` / `LoRA-rand-z` (no task-specific Stage-1 heads); raw outputs in [results/eval_ood/](results/eval_ood/). The CoT-prompt PQ comparison lives in [results/eval_pq_dense_z/](results/eval_pq_dense_z/). Goal-reaching scoring (the meaningful BW metric — simulate the model's plan from initial state, check if `goal_facts ⊆ final_state`) is in [src/score_ood.py](src/score_ood.py); the strict exact-match metric drastically under-estimates BW correctness — see [docs/ood_datasets.md § 2 Scoring](docs/ood_datasets.md).

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
| **BW** | `<PLAN:PICKUP/PUTDOWN/STACK/UNSTACK>` + `<PLAN:ANS>` | PlanBench gold plans (records 200-449) | 250 |
| **PQ** | `<PLAN:DERIVE_TRUE/DERIVE_FALSE>` + `<PLAN:ANS>` | Oracle-generated proofs from forward chaining | 250 |

SFT: Qwen2.5-14B + LoRA r=16, DDP-2 gloo (each task on 2 GPUs in parallel) — total ~7 min wall for all 3.

#### Results

Numbers are in the **Headline results** section above (PT-SFT row). Three distinct failure modes:

- **PQ — format learning, not reasoning**: PT-SFT *underperforms* base (52.5 vs 60). Generations have correct planning-token format but mostly default to "Answer: A" regardless of input. The model learned to mimic the trajectory shape, not the deduction.
- **BW — surface-pattern fit, not planning**: PT-SFT 94.5% is not evidence of compositional planning. PlanBench BW has (a) only 4 action types, (b) short repetitive gold plans, (c) lenient goal-reaching scoring. 250 SFT examples are enough to fit the surface pattern. PQ + GC under the same recipe don't carry, because they need real deduction / combinatorial search.

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
| Qwen-2.5-14B SFT (token-matched) | 0.16 | 1 greedy decode |
| Qwen-2.5-14B ToT top-1 | 0.01 | 5 × (propose + 3 × value) |
| **Qwen-2.5-14B HypPlan DAgger noz** | **0.57** | **1 greedy decode** |
| Qwen-2.5-14B HypPlan DAgger z | 0.55 | 1 greedy decode |
| Qwen-2.5-14B HypPlan + LoRA-G24 (task-z) on G24-100 | 0.12 | 1 greedy decode |

At 14B scale, HypPlan's DAgger-trained LoRA (top-1 greedy) reaches 0.57 with 1 greedy decode. GPT-4 sits ~17 pp above our best 14B number.

### Cross-task (G24-trained LoRA + per-task heads, plus in-domain baselines)

This is the "does the methodology transfer / does it work in-domain on other tasks" picture. PT-SFT and HypPlan-indom train per-task; LoRA-G24 rows reuse the *same* G24-trained LoRA across all OOD tasks.

| Condition | G24-100 | PQ-100 | BW (goal) | GC-100 |
|---|---|---|---|---|
| base (no LoRA, no z) | 11 | 60 | 41 | 61 |
| LoRA-G24 (no z) — original test prompt | 9 | 63 | 43 | 68 |
| LoRA-G24 (rand z) — original test prompt | 4 | 43 | 35 | 60 |
| LoRA-G24 (no z) — CoT prompt | – | 74 | – | – |
| LoRA-G24 (task z, dense per-step) — CoT prompt | – | 67 | – | – |
| PT-SFT in-domain (per-task) | 6 | 52.5 | **94.5\*** | 64 |
| ToT (top-1) | 1 | 41 | 58 | 34 |
| **HypPlan Stage-1+2 in-domain (per-task)** | 12† | **75** | **67** | **88** |

\*PT-SFT BW=94.5% is memorization on PlanBench gold (same distribution as test). Not evidence of compositional planning. See [docs/ood_datasets.md §5](docs/ood_datasets.md#5-planning-tokens-pt-sft-baseline).
†G24 in-domain Stage-1+2 number is from the varied-target setup (also reported as the in-domain Stage-1+2 line). The fixed-target G24 setup gives 0.55–0.57 in the Game-24 table — different LoRA + different test set.

Reading the table:
- **Dense per-step z does NOT help PQ even with a matching CoT prompt** — `LoRA-G24 (no z) — CoT prompt` 74% vs `LoRA-G24 (task z, dense per-step) — CoT prompt` 67% (−7 pp). The geometric z signal genuinely doesn't transfer to deductive reasoning.
- **The G24-trained LoRA itself transfers** to G24-similar tasks even without z: GC +7 pp, PQ +3 pp, BW neutral. The CoT-prompt PQ row (74%) shows the prompt change alone gets most of the in-domain gain — the per-task Stage-1+2 LoRA only adds +1 pp on top.
- **Random z is universally harmful** — z channel is "active" but uninformed-z degrades output.
- **Stage-1+2 in-domain training wins everywhere** (+15 / +26 / +27 pp over base on PQ / BW / GC). BW jumped 10 → 67 between v2 and v3 once we replaced global-min sync with cyclic-pad and added 3 rollouts/problem. Details in [docs/ood_datasets.md §6](docs/ood_datasets.md#6-hypplan-stage-12-trained-in-domain-per-task).

Ablation details are in *Experiments* below.

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
- [src/prompt_builders.py](src/prompt_builders.py) — `sft_prompt_24`, `fewshot_chat_prompt_24`. Each builder returns `(text, add_special_tokens)` so chat-template prompts don't double-add BOS.
- [src/dagger_rollout.py](src/dagger_rollout.py) / [src/train_stage2_dagger.py](src/train_stage2_dagger.py) / [src/generate_24_stage2.py](src/generate_24_stage2.py) thread `prompt_builder` through rollout, training and inference. Config key: `training.prompt_style` (`"sft"` or `"fewshot"`).

---

## Experiments

All experiments below use `Qwen/Qwen2.5-14B-Instruct` as the frozen base model unless stated otherwise, 100 held-out Game-24 problems, seed 1234 (n=1 unless stated), 2-GPU DDP.

### 1. HypPlan (stage 1 + stage 2) on Game-24

- Stage-1 head: `configs/head_qwen14b_origin_ranking.yaml`.
- Stage-2 DAgger: `configs/stage2_dagger_qwen14b_stable.yaml` (lr=5e-5, clip=0.3).
- Artifacts: `checkpoints/head_qwen14b_origin_ranking/`, `results/dagger_stage2_qwen14b_stable/{z,noz}_s1234/`.

**Numbers (lr=5e-5, clip=0.3):**

| Run | Accuracy |
|---|---|
| noz | **0.57** |
| z | 0.55 |

At n=1 seed, apples-to-apples, `noz` beats `z` by 2 pp — within plausible seed noise. The **DAgger loop alone** (no z, no head, no `UpProjector`) gives the best Game-24 accuracy we've measured on Qwen-14B. A 3-seed rerun of `{noz, z}` is the next cheap experiment to make this crisp.

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

**Result (seed 1234, 2-GPU DDP, 3 epochs, 408 optim steps, ~12 min):** Qwen-14B SFT = **0.16**, vs Qwen-14B HypPlan DAgger noz = **0.57** (+41 pp at identical LoRA / prompt). The gain isn't coming from the adapter or extra supervised tokens — it's the DAgger rollout-and-relabel loop.

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

---

## Planned: real ToT on 4 missing datasets

Real ToT (paper-faithful Yao et al. 2023 BFS: propose → evaluate → select-top-5) is **already implemented** for G24 ([src/tot_baseline.py](src/tot_baseline.py)) and PQ/BW/GC ([src/tot_ood.py](src/tot_ood.py)). The 4 newer datasets — **rulechain, CLUTRR, Number-path, ProofWriter** — currently have no real ToT result. Earlier "ToT" numbers in these cells came from a wrong-runner shortcut (K-sampled structured greedy, no propose/evaluate/select) and were deleted on 2026-04-27.

### Plan

1. Extend [src/tot_ood.py](src/tot_ood.py) with 4 new task adapters:
   - `RulechainAdapter` — propose: next forward-chaining rule application; value: `sure/likely/impossible` for the target predicate's reachability.
   - `ClutrrAdapter` — propose: next kinship composition step; value: `sure/likely/impossible` for the target relation.
   - `NumpathAdapter` — propose: next op application from the per-problem op set; value: heuristic distance from current value to target.
   - `ProofwriterAdapter` — propose: next inference (apply rule + facts → new fact); value: `sure/likely/impossible` that the target derives.
2. Hyperparameters fixed across all 4: `n_generate=1, n_evaluate=3, n_select=5, T=0.7`, max_depth tuned per task (rulechain/proofwriter: 8; numpath: 6; clutrr: 4).
3. Run 6-GPU sharded eval (same pattern as `scripts/run_one_baseline.sh`). Expect ~50-100× the per-problem compute of greedy → several GPU-hours per task.
4. Once results land, fill in the four `TBD` cells in [docs/HANDOFF.md](docs/HANDOFF.md) and re-commit.

### Remove the wrong-mode runner

[src/eval_baseline_kpath.py](src/eval_baseline_kpath.py)'s `--mode tot` should be removed (or renamed to `--mode structured-greedy`) so the mislabeling can't recur.

---

## Planned: efficiency comparison (HypPlan vs baselines on Game-24)

Quantify inference cost of HypPlan vs baselines on a single GPU. Three metrics per system, measured on the same 100-problem Game-24 test set:

| Metric | What it counts |
|---|---|
| **LLM forward passes** | Total `model.forward(...)` calls per problem (decode steps × samples / branches) |
| **Generated tokens** | Total assistant tokens emitted per problem |
| **Latency (s)** | Wall-clock from start of decode to final answer parsed, median over 100 problems |

### Systems to compare

- **Qwen-14B fewshot (no LoRA)** — 1 greedy decode
- **Qwen-14B SFT (token-matched)** — 1 greedy decode
- **HypPlan DAgger (z, noz)** — 1 greedy decode + ~5 head-forward passes (one per step boundary; head is tiny relative to base)
- **ToT (top-1)** — `n_generate=1, n_evaluate=3, n_select=5, T=0.7`
- **Self-Consistency (majority)** — K=5 samples, T=0.7

### Setup

- Single A6000 (48 GB), single process, batch size 1, no DDP, seed 1234.
- Run on a quiet GPU to avoid contention.

### Implementation sketch

Modify `src/eval_baseline_kpath.py` and `src/generate_24_stage2.py` to log per-problem:
- A counter wrapped around `model(...)` to record forward-pass count.
- `output_ids[input_len:].shape[0]` summed across all decode calls (including ToT propose / value sub-calls).
- `time.perf_counter()` around the whole-problem decode.

Append the resulting "Inference compute" table to *Headline results* once collected.
