# HypPlan: Tree-Distortion Hyperbolic Planning for LLM Reasoning

A two-stage framework. **Stage 1** teaches a small head to embed reasoning-tree states into a hyperbolic space so that `|z|` tracks solution-proximity — pure geometric supervision on an enumerated state tree, no language-model loss. **Stage 2** trains a fresh LoRA + `UpProjector` on top of the frozen SFT base using **DAgger with a tree oracle**: the current policy rolls out trajectories freely, the oracle labels winning ops at each reached state, and CE trains the LoRA on those labels. The frozen head's geometric `z` is injected as a virtual token before each step boundary.

Runs on **Game of 24**; the Countdown port (N=6 pool, variable integer target) is in progress. The MATH pipeline from the original HypPlan also lives in this tree (see `src/train_stage1.py` etc.) but the active pipeline is the two-stage Game-of-24 flow documented below.

---

## The two stages

### Stage 1 — hyperbolic head (LLM frozen)

For each Game-of-24 problem we enumerate the full state tree (root = initial 4 numbers, children = all legal (a, op, b) applications, leaves = 1-number terminal states). Each node's state text is encoded by the frozen SFT-merged Llama-3.1-8B. A small head MLP maps that hidden vector to a low-dim point (default `hyp_dim=32`) in a hyperbolic space via exp-map at the origin.

**Loss: `origin_ranking`.** A margin hinge on distance-to-origin, with target `v(s)` = BFS edge distance from state `s` to the nearest success leaf in the enumerated tree. For any sampled pair `(s_i, s_j)` with `v(s_i) < v(s_j)`:

`L = max(0, d_H(z_i, 0) − d_H(z_j, 0) + margin)`

This makes `|z|` track solution-proximity: states closer to a solution are pulled toward the origin, states farther are pushed outward. Supported manifolds: Poincaré ball and Lorentz hyperboloid (plus a Euclidean variant reserved for the geometry ablation — see *Hyperbolic vs Euclidean* below).

### Stage 2 — DAgger with tree oracle (base and head frozen)

A fresh LoRA adapter on the SFT-merged base + a small `UpProjector` (lifts the 32-dim hyperbolic point back to hidden_dim=4096) are the only trainable parts. Training uses **DAgger** (expert iteration, AlphaGo-style):

At each epoch, for each training problem:
1. **Rollout under current policy** — generate step-by-step with T=0.7, top-p=0.95, injecting `z_t` as a virtual token at each step boundary (z run only; the no-z control run skips the injection). Continue until a valid solution, an invalid step, or step budget exhausted.
2. **Oracle labeling** — for each step-boundary state reached, query the oracle (memoized recursive search via `src/oracle_24.py`): given `remaining`, return all ops whose resulting state can still reach 24.
3. **Invalid-step handling** — if the model emits a step with wrong arithmetic or hallucinated operands, truncate the trajectory at that step. Earlier valid states still contribute.
4. **Training pass** — for each collected (state, z, winning_ops) tuple, pick one winner (lex tiebreak) and CE-train the model to emit its full step text. Backprop into LoRA + UpProjector; head and base stay frozen.

The canonical state text for each boundary passes through frozen base + frozen head → `z`; up-projector produces `z_inj` injected before the next step's tokens. Loss is single-winner CE (phase 1); phase-2 upgrade would be log-of-sum over all winners' step-text likelihoods.

---

## Pipeline end-to-end

### 0. Setup

```bash
pip install -r requirements.txt
```

**Hardware**: 8× NVIDIA A6000 (48 GB). GPUs 5↔7 have a broken NCCL pair on this node; training scripts default to `MEM_THRESHOLD=30000`-MiB auto-detect which usually picks a safe trio.

### 1. Data preparation — tree cache

Enumerates the full state tree for every solvable 24-problem in `data/24_{train,val,test}.jsonl` and caches (a) the tree metadata (`parents`, `depths`) and (b) the frozen SFT LLM's last-token hidden state for every node as float16 `.npy` memmaps.

```bash
bash scripts/run_gen_tree_data.sh
```

- Sharded across all detected free GPUs (one python process per GPU), each shard handles `idx % world == rank` problems.
- Resume-safe: existing `data/trees/{split}/problem_{idx}.pt` + `hidden_{idx}.npy` files are skipped.
- Produces ~33 GB: `data/trees/{train,val,test}/` — 1090 train / 136 val / 136 test trees, ~3000 nodes per tree on average.

### 2. Stage 1 — head training + evaluation

```bash
bash scripts/run_train_head.sh poincare origin_ranking
```

Each run trains for 20 epochs on the cached hidden states (no LLM loaded), saves `checkpoints/head_poincare_origin_ranking/head.pt`, then runs `src.eval_head` to produce:

- `results/head_eval/poincare_origin_ranking/metrics.json` — Spearman rank correlation of `|z|` vs `v(s)`, origin-distance histograms (val + test).
- `results/head_eval/poincare_origin_ranking/vis_tree_{idx}.png` — 2D tangent-PCA visualization of example trees.

### 3. Stage 2 — DAgger training + inference + eval

```bash
# Per-run per-seed launcher. Requires a stage-1 head first.
bash scripts/run_train_stage2_dagger.sh <noz|z> poincare_origin_ranking [seed]
```

Each invocation trains one run (z-injected or no-z control) for one seed. A full 3-seed two-run sweep = 6 invocations. The driver (a) trains the LoRA + UpProjector across all detected free GPUs with DDP (manual gradient averaging — see *Distributed training notes* below), (b) generates 100 test-problem solutions with `src.generate_24_stage2`, (c) validates them with `src.evaluate_24`.

Artifacts:
- `checkpoints/dagger_stage2_{head_tag}/{noz|z}_s{seed}/` — LoRA + UpProjector.
- `results/dagger_stage2_{head_tag}/{noz|z}_s{seed}/{generations.jsonl, metrics.json, rollout_stats_epoch*.json}`.

No-z is the control (`--use_z` off). Inference-time `--random_z` is also supported via `src.generate_24_stage2` for sanity-checking a trained z-run checkpoint.

---

## Evaluation

`src.evaluate_24` validates each generated 3-step solution: parses `Step N: a op b = r`, replays the arithmetic, checks that all 4 input numbers are used exactly once, and confirms the final result is 24. Accuracy = fraction of problems with a valid 3-step solution.

All runs default to 100 held-out test problems from `data/24_test_tot.jsonl`. Scale via `--limit`.

---

## Results so far (100 problems, greedy decoding, ≤3 z-injections)

All Stage-2 numbers below are reported as **mean ± stdev across 3 DDP seeds
(1234, 4242, 6666)** on 2-GPU DDP.

| System | Accuracy | Notes |
|---|---|---|
| SFT-only baseline | 0.12 | `results/24_sft_tot/` |
| **Stage-2 no-z (control, 3-seed mean)** | **0.333 ± 0.019** | `results/dagger_stage2_poincare_origin_ranking/noz_s*/` |
| **Stage-2 + Poincaré z (3-seed mean)** | **0.410 ± 0.020** | `results/dagger_stage2_poincare_origin_ranking/z_s*/` |
| **Stage-2 + Euclidean z (3-seed mean)** | 0.330 ± 0.090 | `results/dagger_stage2_euclidean_origin_ranking/z_s*/` |

Per-seed raw numbers (DDP):

| Seed | noz | z (Poincaré) | Δ_hyp | z (Euclidean) | Δ_euc |
|---|---|---|---|---|---|
| 1234 | 0.32 | 0.43 | +11 | 0.23 | **−9** |
| 4242 | 0.32 | 0.41 | +9 | 0.35 | +3 |
| 6666 | 0.36 | 0.39 | +3 | 0.41 | +5 |

---

## Why not teacher forcing?

Our initial Stage-2 design trained the LoRA on **teacher-forced** trajectories
— injecting `z` at step boundaries and optimizing standard next-token CE. That
run (preserved under `results/hyp_stage2_*`) landed at 0.21 accuracy —
statistically indistinguishable from a null baseline (random `z`). Two
compounding reasons:

1. **Teacher forcing eliminates the uncertainty z was designed for.** At each
   step boundary during training, the model is conditioned on the *correct*
   preceding trajectory. z — a compressed summary of that same trajectory —
   is informationally redundant given the context the LLM already has. CE has
   no gradient pressure to extract z's content.
2. **CE does not reward z-usage.** The LM can drive CE low via alternative
   paths (preceding text, base SFT priors). Nothing forces the policy to
   *depend* on z.

Evidence from our null-baseline experiments: LoRA trained with real z broke
when given random z at test (the "+9pp" figure we initially misinterpreted),
and LoRA trained with random z worked normally with random z. That pattern
means the LoRA learned z's **distributional statistics** (norm, variance),
not its **semantic content** — it used z as a calibration signal, not a
payload.

**DAgger fixes this** by treating the head as a privileged critic (Stage 1
had access to the enumerated tree — solution locations, distance to nearest
success) and the LoRA as a policy trained under its own state distribution.
Under free generation the model reaches genuinely uncertain states; z then
carries decision-relevant information the model cannot trivially recompute
from context. See the Stage-2 section above for the full training loop.

### Two-run experimental design

Both runs use **identical** code path, warm start, sampling hyperparams,
oracle rules, and DAgger schedule. A single `--use_z` flag toggles z-injection
on/off. This isolates z's contribution from the exposure-bias fix, both of
which independently should raise accuracy. The clean metric:
`Δ_accuracy = acc(z run) − acc(no-z run)`.

### Warm start

Warm start from:
- SFT-merged base (frozen) — already hits 0.12 accuracy.
- **Fresh LoRA** with standard PEFT init (A ∼ 𝒩, B = 0, so delta = 0 at
  step 0).
- **Small-std-init UpProjector** (σ=1e-3 on the final Linear's weight, bias=0).
  We initially tried fully zero-init but `LayerNorm(0)` combined with the
  Llama-3.1-Instruct chat template triggered a degenerate fallback where the
  model emitted `"assistant\n..."` at step 1. A tiny non-zero init sidesteps
  this while staying close to "no effect" at step 0.
- Frozen `head_{manifold}_origin_ranking` as the critic.

First rollout with this init ≈ pure SFT-merged behavior (0.12), without
inheriting any bad z-attention habits. DAgger teaches the LoRA to use z
from scratch.

### Decisions locked in

1. Drop invalid trajectories from the invalid step onward. Log drop rate per
   epoch; alarm if >50% after epoch 0.
2. Single-winner CE (phase 1, lex tiebreak). See loss note above.
3. Fresh LoRA (B=0) + small-std-init UpProjector (σ=1e-3). See warm-start note.
4. T=0.7, top-p=0.95 for rollout. Greedy for eval.
5. Lockstep DAgger: per epoch, rollout all 1090 train problems (3 trajectories
   each ≈ 3300 trajectories), then one CE pass over collected pairs. Repeat
   for 3 epochs. Both runs execute in parallel.

### Components (files)

- `src/oracle_24.py` — Given `remaining`, returns winning next-ops via a
  memoized recursive search (not a tree-file lookup). Handles any state
  the model can reach, including off-tree sequences.
- `src/dagger_rollout.py` — One-problem rollout: token-by-token sampling
  with per-step z injection, step parsing, oracle labeling, invalid-step
  detection, tolerant regex for z-injection prefix artifacts.
- `src/train_stage2_dagger.py` — Two-run Stage-2 (DAgger) trainer. `--use_z`
  flag, `--seed` override for multi-seed runs. Manual gradient averaging
  under DDP. NCCL collective timeout raised to 60 min to tolerate rank
  imbalance during variable-length rollouts.
- `configs/stage2_dagger.yaml` — Stage-2 config template.
- `scripts/run_train_stage2_dagger.sh` — Per-run per-seed launcher:
  `bash run_train_stage2_dagger.sh <noz|z> <head_tag> [seed]`.

Artifacts:
- `checkpoints/dagger_stage2_{head_tag}/{noz|z}_s{seed}/` — LoRA +
  UpProjector.
- `results/dagger_stage2_{head_tag}/{noz|z}_s{seed}/{generations.jsonl, metrics.json, rollout_stats_epoch*.json}`.

See [docs/dagger_walkthrough.md](docs/dagger_walkthrough.md) for a concrete
example walkthrough (rollout terminology, oracle mechanics, z vs no-z run
side-by-side on problem `4,5,6,10`).

---

## Project layout (v2 files only)

```
HypPlan/
├── configs/
│   ├── head.yaml              # stage-1 template (manifold switchable; loss = origin_ranking)
│   └── stage2_dagger.yaml     # stage-2 (DAgger) template
├── src/
│   ├── tree_data.py           # enumerate_tree, render_state, pair_distances_lca
│   ├── hyperbolic.py          # Lorentz ops (unchanged from v1)
│   ├── head.py                # HyperbolicHead (Poincaré/Lorentz/Euclidean) + UpProjector
│   ├── train_head.py          # stage-1 trainer (origin_ranking loss)
│   ├── eval_head.py           # Spearman(|z|, v(s)) + 2D viz
│   ├── oracle_24.py           # stage-2 oracle: winning_ops(remaining)
│   ├── dagger_rollout.py      # stage-2 rollout + oracle labeling
│   ├── dataset_24_stage2.py   # per-boundary canonical state tokenization
│   ├── train_stage2_dagger.py # stage-2 (DAgger) trainer (two-run, DDP)
│   ├── generate_24_stage2.py  # inference (supports --no_z_inject, --random_z)
│   └── evaluate_24.py         # solution validator (unchanged)
├── data/
│   ├── generate_tree_data.py  # offline tree + hidden-state cache builder
│   ├── 24_{train,val,test}.jsonl
│   └── trees/                 # cached tree metadata + hidden states
├── docs/
│   └── dagger_walkthrough.md  # concrete example of stage-2 mechanics
├── scripts/
│   ├── run_gen_tree_data.sh
│   ├── run_train_head.sh
│   └── run_train_stage2_dagger.sh     # stage-2 per-run per-seed
├── checkpoints/
│   ├── sft_24_tot_merged/             # frozen feature extractor
│   ├── head_{manifold}_origin_ranking/ # stage-1 heads
│   └── dagger_stage2_{head_tag}/{noz|z}_s{seed}/
└── results/
    ├── head_eval/{manifold}_origin_ranking/
    └── dagger_stage2_{head_tag}/{noz|z}_s{seed}/
```

Old v1 files (`train_plan_24.py`, `generate_24_plan.py`, `train_sft_24.py`, `train_stage1.py`, `train_stage2.py` — the teacher-forced precursor, …) remain in place as reference — not deleted so prior `results/` stay reproducible.

---

## Distributed training notes

Stage-2 (DAgger) DDP uses **manual gradient averaging** rather than `torch.nn.parallel.DistributedDataParallel`:

- Seed `torch.manual_seed(1234)` before LoRA + `UpProjector` init so every rank gets identical weights without a broadcast collective.
- After `loss.backward()`, iterate over trainable params and call `dist.all_reduce(p.grad, op=SUM) ; p.grad.div_(world_size)` before `optimizer.step()`.

Why not standard DDP? Stage-2's computation graph changes per iteration (variable-K per-boundary inner loop, plus `disable_adapter()` sub-forwards for state encoding). That makes DDP's bucket-ready ordering diverge across ranks and deadlock the first auto-reduce. Manual averaging sidesteps the problem; the sync cost is trivial for our ~22M trainable params.

NCCL topology gotcha on this host: GPUs 5↔7 are a broken pair at the NCCL level (works pair-wise with other GPUs; deadlocks when both are in the same process group). If you must use all 8 GPUs, verify with `scripts/test_nccl.sh`-style probe first.

---

## Roadmap and progress

Short-term decisions tracked here so context isn't lost across sessions.

### Current focus: rebuild on Qwen-2.5-14B-Instruct as the shared base

After running few-shot validation on several candidates:

| Model (greedy + 3-shot Game-24) | Accuracy | Format rate |
|---|---|---|
| Llama-3.1-8B-Instruct | 3% | 96% |
| Qwen-2.5-7B-Instruct | 0% (verified earlier by user) | — |
| Llama-3.1-8B-SFT (our trained checkpoint) | 12% | ~100% |
| **Qwen-2.5-14B-Instruct** | **11%** | **100%** |

Qwen-2.5-14B-Instruct matches SFT-8B accuracy at zero SFT cost and gives a
clean "same base as ToT" story. The pivot: both HypPlan and the ToT baseline
will use `Qwen/Qwen2.5-14B-Instruct` as the frozen base. This lets us skip
the SFT preprocessing step and compare like-for-like against ToT.

Few-shot validator: [scripts/fewshot_baseline.py](scripts/fewshot_baseline.py)
+ [scripts/run_fewshot_baseline.sh](scripts/run_fewshot_baseline.sh).

### Three experiments (1 seed each, multi-seed comes later)

All three use `Qwen/Qwen2.5-14B-Instruct` as the frozen base model.

Code plumbing that supports all three:
- [src/prompt_builders.py](src/prompt_builders.py) — `sft_prompt_24`,
  `fewshot_chat_prompt_24`, `sft_prompt_cd`, `fewshot_chat_prompt_cd`.
  Each builder returns `(text, add_special_tokens)` so chat-template
  prompts don't double-add BOS.
- `src/train_head.py` now supports `origin_ranking_rank` loss (scale-
  invariant rank-based margin for wide-range Countdown v).
- `src/dagger_rollout.py` / `src/train_stage2_dagger.py` /
  `src/generate_24_stage2.py` thread `prompt_builder` through rollout,
  training and inference. Config key: `training.prompt_style`
  (`"sft"` or `"fewshot"`).

**1. Qwen-14B + HypPlan (stage 1 + stage 2) on Game-24.**
   - Tree cache: `bash scripts/run_gen_tree_data.sh` with
     `BASE_MODEL=Qwen/Qwen2.5-14B-Instruct OUT_DIR=data/trees_qwen14b
     LOG_DIR=logs/gen_tree_qwen14b BATCH_SIZE=32`. In progress; 1090
     train + 136 val + 136 test problems.
   - Stage-1 head: `configs/head_qwen14b_origin_ranking.yaml` →
     `bash scripts/run_train_head.sh poincare origin_ranking` after
     pointing BASE_CONFIG to the new config, or invoke
     `python -m src.train_head --config configs/head_qwen14b_origin_ranking.yaml`
     + `python -m src.eval_head --config ...`.
   - Stage-2 DAgger: `configs/stage2_dagger_qwen14b.yaml` (sets
     `prompt_style: fewshot`). Launch:
     `BASE_CONFIG=configs/stage2_dagger_qwen14b.yaml
      RUN_CONFIG=configs/stage2_dagger_qwen14b_run.yaml
      bash scripts/run_train_stage2_dagger.sh z qwen14b_origin_ranking 1234`
     (and similarly for `noz`).
   - Artifacts: `checkpoints/head_qwen14b_origin_ranking/`,
     `checkpoints/dagger_stage2_qwen14b_origin_ranking/{z,noz}_s1234/`,
     `results/dagger_stage2_qwen14b_origin_ranking/{z,noz}_s1234/`.

**2. Qwen-14B + ToT on Game-24.**
   - `bash scripts/run_tot_qwen14b.sh` — single-model ToT
     (`--shared_model --use_chat_template`, paper-default
     `n_generate=1, n_evaluate=3, n_select=5, T=0.7`).
   - Reports both any-of-top-5 (matches ToT paper's 74% metric) and
     top-1 accuracy. Overlap with `data/24_test_tot.jsonl` is 100/100.
   - Artifacts: `results/tot_baseline_qwen14b/seed_1234/`.

**3. Qwen-14B + HypPlan (stage 1 + stage 2) on Countdown.**
   - Tree cache (Qwen Countdown): `bash scripts/run_gen_tree_data_cd.sh`
     with `BASE_MODEL=Qwen/Qwen2.5-14B-Instruct
     OUT_DIR=data/cd_trees_qwen14b
     LOG_DIR=logs/gen_tree_cd_qwen14b`.
   - Stage-1 head: `configs/head_cd_qwen14b_rank.yaml`
     (`loss: origin_ranking_rank` — scale-invariant rank-based margin).
     Launch `python -m src.train_head --config
     configs/head_cd_qwen14b_rank.yaml`.
   - Stage-2 DAgger: `configs/stage2_dagger_cd_qwen14b.yaml`. Launch
     `bash scripts/run_train_stage2_dagger_cd.sh z cd_qwen14b_rank 1234`.
     Code: [src/dagger_rollout_cd.py](src/dagger_rollout_cd.py) (integer
     arithmetic, variable target, 5 steps, `CountdownOracle` per problem),
     [src/train_stage2_dagger_cd.py](src/train_stage2_dagger_cd.py),
     [src/generate_cd_stage2.py](src/generate_cd_stage2.py).
   - Gate: after epoch-0 rollout, check stats['n_boundaries_invalid'] /
     stats['n_boundaries_total']. If >50%, SFT-free bootstrap is too
     weak — fall back to SFT'd Countdown base.

### Completed so far (Game-24 legacy, Llama-based)

Preserved for reference; the paper narrative will use the Qwen-14B numbers
once they land.

| System | Accuracy | Base | Notes |
|---|---|---|---|
| SFT-only | 0.12 | Llama-3.1-8B + SFT | `results/24_sft_tot/` |
| Stage-2 DAgger no-z (control) | 0.333 ± 0.019 | Llama-3.1-8B + SFT | 3-seed DDP, kept as historical |
| Stage-2 DAgger + Poincaré z | 0.410 ± 0.020 | Llama-3.1-8B + SFT | 3-seed DDP, kept as historical |

### Earlier focus (superseded): Game-24 ToT baseline

**Goal:** add a competitive search-based baseline at matched base-model
capability (Llama-3.1-8B, not GPT-4), so our 0.41 HypPlan number has a
fair point of comparison on the same 100 test problems (which are exactly
the ToT paper's 900-999 split — verified).

**Setup:**
- Generator = `checkpoints/sft_24_tot_merged` (our SFT; matches HypPlan's base).
- Evaluator = `meta-llama/Llama-3.1-8B-Instruct` (base, no SFT — our SFT
  distribution can't emit `sure/likely/impossible` labels).
- ToT hyperparams: paper-default (`n_generate_sample=1`,
  `n_evaluate_sample=3`, `n_select_sample=5`, `T=0.7`, BFS depth 3).
- Prompts: paper-verbatim from
  [ToT repo](https://github.com/princeton-nlp/tree-of-thought-llm).
- Dual-format parser: accepts either ToT format (`"a + b = c (left: x y z)"`)
  or SFT-leaked format (`"Step N: a + b = c. Remaining: x y z"`).
- Report: both best-of-5 (any-of-final-beam correct; matches the paper's
  74%) and best-of-1 (top-score only; matches our greedy eval convention).
- Seeds: 3 for stability.

**Companion run for fairness:** also run HypPlan best-of-5 (T=0.7, 5
attempts, any-correct) on the same 100 problems so the "best-of-5 column"
is apples-to-apples between ToT and HypPlan.

**Files to create:**
- `src/tot_baseline.py` — BFS runner (two-model setup, staged loading).
- `scripts/run_tot_baseline.sh` — launcher.
- `results/tot_baseline/{seed_*}/{generations.jsonl, metrics.json}`.

### After Game-24 ToT lands: Countdown continuation

Countdown port is in progress. Data and SFT pipelines are done; stage-1 and
stage-2 still need to be wired.

**Countdown — done:**
1. Oracle (`src/oracle_cd.py`) with variable target, integer ops
   (non-negative subtraction, exact division); offline cache at
   `data/cd_oracle_cache/` (18 MB).
2. Problem generator (`data/generate_countdown.py`) — 1000/100/100 solvable
   problems with N=6 pool (5 small ∪ 1 big) and target ∈ [100, 999].
3. SFT trajectories (`data/generate_cd_trajectories.py`, `data/cd_*_sft.jsonl`).
4. Countdown SFT (`src/train_sft_cd.py`, `configs/sft_cd.yaml`,
   `scripts/run_sft_cd.sh`) trained with QLoRA 5 epochs 2-GPU DDP; baseline
   **1% accuracy** (matches the SOS literature's 1-5% range).
5. Merged checkpoint (`checkpoints/sft_cd_merged`, 15 GB).
6. Tree-data generation (`src/tree_data_cd.py`,
   `data/generate_tree_data_cd.py`, `scripts/run_gen_tree_data_cd.sh`) —
   `data/cd_trees/` (7.6 GB, 1000+200 trees with hidden states).
7. v-value redefined to **continuous `|final_value − target|`** (every state
   gets a finite v, target-reachable states get v=0). `compute_v_values` in
   `src/tree_data_cd.py` now enumerates the full DAG from root (not from
   the oracle cache, which is incomplete because of can_reach early-return).

**Countdown — pending:**
1. Clean up v-values on the test split. Train/val got updated in place by
   `data/recompute_v_values.py`; test failed because the guided-trajectory
   logic change altered tree topology from what the saved hidden states
   were built for. Options: (a) regenerate all trees from scratch (~1 hr
   GPU), or (b) accept that stage-1 only trains on train+val and test is
   eval-only with no v-value dependence.
2. Stage-1 head training for Countdown: port `src/train_head.py` to read
   `data/cd_trees/`, sample pairs from the new continuous v distribution
   (reachable-aware), handle the scale (use `log(1+v)` or rank-based
   margin to keep the hinge meaningful when v ranges 0–1M).
3. Stage-1 evaluation: Spearman(|z|, v(s)), 2D viz adapted to Countdown.
4. Stage-2 (DAgger) port for Countdown:
   - Wire `src/oracle_cd.py` into `src/dagger_rollout.py` as the
     target-aware oracle (current rollout uses `oracle_24` hard-coded).
   - Integer-arithmetic rollout parser (operands can be 3+ digits, no
     negative intermediates, exact division).
   - New config `configs/stage2_dagger_cd.yaml`: `max_z_injections=5`,
     `max_seq_len≥384`, target in prompt.
   - EOS enforcement during rollout generation: stop at 5 parsed step
     boundaries; otherwise SFT's 6-8-step drift will force-invalid most
     trajectories.
   - **Gate:** before full DAgger run, measure the fraction of rollout
     steps that land in the game's valid state space (parseable +
     operand-correct) on a smoke run. If that fraction is low even after
     EOS enforcement, we need a stronger SFT base first.
5. Stage-2 DAgger training (3 seeds, DDP); stage-2 eval and compare.
6. (Optional) ToT-style baseline adapted for Countdown, if the method
   transfers and we want the same fair comparison.

### Open research questions / backlog

- **Seeds for statistical significance.** Current hyperbolic-vs-Euclidean
  gap is +7.7pp on 3 seeds (p ≈ 0.34). For publication-grade p<0.05 we'd
  want n ≥ 5, ideally 10. Cheap to add.
- **Phase-2 loss upgrade.** Replace single-winner CE with log-of-sum over
  full step texts of all winners (≈K× compute). Not needed for current
  numbers but cleaner theoretically.
- **Approximate oracle for N > 6.** Exact oracle scales poorly past N=7
  even in C++. Approximate critic (beam/MCTS or learned value) would be
  the path to larger Countdown or other planning domains.
