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
1. **Rollout under current policy** — generate step-by-step with T=0.7, top-p=0.95, injecting `z_t` as a virtual token at each step boundary (z-arm only; the no-z control arm skips the injection). Continue until a valid solution, an invalid step, or step budget exhausted.
2. **Oracle labeling** — for each step-boundary state reached, query the oracle (memoized recursive search via `src/oracle_24.py`): given `remaining`, return all ops whose resulting state can still reach 24.
3. **Invalid-step handling** — if the model emits a step with wrong arithmetic or hallucinated operands, truncate the trajectory at that step. Earlier valid states still contribute.
4. **Training pass** — for each collected (state, z, winning_ops) tuple, pick one winner (lex tiebreak) and CE-train the model to emit its full step text. Backprop into LoRA + UpProjector; head and base stay frozen.

The canonical state text for each boundary passes through frozen base + frozen head → `z`; up-projector produces `z_inj` injected before the next step's tokens. Loss is single-winner CE (phase 1); phase-2 upgrade would be log-of-sum over all winners' step-text likelihoods.

**Why DAgger and not teacher forcing?** Our initial teacher-forced attempt landed at null-baseline accuracy because `z` was informationally redundant with the teacher-forced context. See *Why not teacher forcing?* below.

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
# Per-arm per-seed launcher. Requires a stage-1 head first.
bash scripts/run_train_stage2_dagger.sh <noz|z> poincare_origin_ranking [seed]
```

Each invocation trains one arm (z-injected or no-z control) for one seed. A full 3-seed two-arm sweep = 6 invocations. The driver (a) trains the LoRA + UpProjector across all detected free GPUs with DDP (manual gradient averaging — see *Distributed training notes* below), (b) generates 100 test-problem solutions with `src.generate_24_stage2`, (c) validates them with `src.evaluate_24`.

Artifacts:
- `checkpoints/dagger_stage2_{head_tag}/{arm_s{seed}}/` — LoRA + UpProjector.
- `results/dagger_stage2_{head_tag}/{arm_s{seed}}/{generations.jsonl, metrics.json, rollout_stats_epoch*.json}`.

No-z ablation is the control arm (`--use_z` off). Inference-time `--random_z` is also supported via `src.generate_24_stage2` for sanity-checking a trained z-arm checkpoint.

---

## Evaluation

`src.evaluate_24` validates each generated 3-step solution: parses `Step N: a op b = r`, replays the arithmetic, checks that all 4 input numbers are used exactly once, and confirms the final result is 24. Accuracy = fraction of problems with a valid 3-step solution.

All runs default to 100 held-out test problems from `data/24_test_tot.jsonl`. Scale via `--limit`.

---

## Results so far (100 problems, greedy decoding, ≤3 z-injections)

All Stage-2 numbers below are reported as **mean ± stdev across 3 DDP seeds
(1234, 4242, 6666)**; 2-GPU DDP is the regime used for all reported Stage-2
values because single-GPU runs showed ~3× higher seed variance (see
*Single-GPU vs DDP* below).

| System | Accuracy | Notes |
|---|---|---|
| SFT-only baseline | 0.12 | `results/24_sft_tot/` |
| **Stage-2 no-z (control, 3-seed mean)** | **0.333 ± 0.019** | `results/dagger_stage2_poincare_origin_ranking/noz_s*/` |
| **Stage-2 + Poincaré z (3-seed mean)** | **0.410 ± 0.020** | `results/dagger_stage2_poincare_origin_ranking/z_s*/` |
| **Stage-2 + Euclidean z (3-seed mean)** | 0.330 ± 0.090 | `results/dagger_stage2_euclidean_origin_ranking/z_s*/` |

**Stage-2 headline (DDP, n=3 seeds):**

| Quantity | Value |
|---|---|
| No-z → with-z Δ (Poincaré), paired | **+7.7 ± 4.2 pp** (+11, +9, +3 per seed) |
| No-z → with-z Δ (Euclidean), paired | −0.3 ± 7.6 pp (−9, +3, +5) |
| SFT → Stage-2 no-z (exposure-bias fix) | +21.3 pp |
| SFT → Stage-2 + Poincaré z (total) | +29.0 pp |

Per-seed raw numbers (DDP):

| Seed | noz | z (Poincaré) | Δ_hyp | z (Euclidean) | Δ_euc |
|---|---|---|---|---|---|
| 1234 | 0.32 | 0.43 | +11 | 0.23 | **−9** |
| 4242 | 0.32 | 0.41 | +9 | 0.35 | +3 |
| 6666 | 0.36 | 0.39 | +3 | 0.41 | +5 |

**Single-GPU vs DDP.** We originally ran seed 1234 with 2-GPU DDP and three
more seeds on single-GPU. The single-GPU regime showed *much* higher
seed variance (noz = 0.18 / 0.21 / 0.35, range 17pp) because batch_size=1
makes per-step gradient noise 2× higher, and the single-GPU run does 2× as
many optimizer steps to process the same data. We therefore rely on the
3-seed DDP numbers as the headline; single-GPU numbers are kept only for
completeness under `results/dagger_stage2_poincare_origin_ranking/{noz_s2024,
z_s2024, noz_s7777, z_s7777, noz_s9999, z_s9999}/`.

---

## Why not teacher forcing? (motivation)

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

### Two-arm experimental design

Both arms use **identical** code path, warm start, sampling hyperparams,
oracle rules, and DAgger schedule. A single `--use_z` flag toggles z-injection
on/off. This isolates z's contribution from the exposure-bias fix, both of
which independently should raise accuracy. The clean metric:
`Δ_accuracy = acc(z-arm) − acc(no-z-arm)`.

### Warm start (critical design choice)

**Do NOT warm-start from the failed teacher-forced checkpoints.** Our
null-baseline evidence shows those LoRAs learned z's distribution as a
calibration signal, not its content. Starting from them would inherit
attention patterns that route *around* z's semantic content — exactly the
local minimum DAgger needs to escape.

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
   for 3 epochs. Both arms run simultaneously.

### Results summary

See the headline table at the top. Key numbers:

- **Stage-2 no-z alone**: 0.333 ± 0.019 — +21pp over the 0.12 SFT baseline.
  This is the exposure-bias fix: training on model-reached states teaches
  recovery behavior that teacher forcing can't.
- **Stage-2 + Poincaré z**: 0.410 ± 0.020 — adds **+7.7 ± 4.2pp** on top,
  z > noz on 3/3 seeds.
- Total lift over SFT: **+29pp** (0.12 → 0.41).

### Hyperbolic vs Euclidean ablation

To test whether the lift requires *hyperbolic* geometry specifically or just
any compressed MLP summary of the hidden state, we swapped the Poincaré head
for a Euclidean variant of identical architecture (same MLP widths, same
`hyp_dim=32`, same `origin_ranking` loss — only the exp-map and distance
function differ; `origin_distance = ‖z‖₂`, pairwise = L2). Same 3 DDP seeds.
The `noz` arm is reused unchanged (it never calls the head).

| Quantity | Poincaré | Euclidean |
|---|---|---|
| z-arm mean ± std | **0.410 ± 0.020** | 0.330 ± 0.090 |
| Δ (z − noz) mean ± std | **+7.7 ± 4.2 pp** | −0.3 ± 7.6 pp |
| Seeds with positive Δ | 3/3 | 2/3 (seed 1234 → −9pp) |

Poincaré beats Euclidean in mean z-arm accuracy by ~8pp with less than half
the variance, and is monotonically positive across seeds while Euclidean's
lift is seed-dependent (one seed catastrophically worse than no-z at all).
Paired t-test on `Δ_hyp − Δ_euc` per seed (+20, +6, −2): t≈1.25, df=2, p≈0.34
— direction supports the geometry claim but **n=3 is insufficient for
p<0.05**.

Mechanistic hypothesis for Euclidean's instability: `origin_ranking` under
L2 is scale-invariant (the margin inequality can be satisfied by arbitrary
|z| scaling). Hyperbolic spaces bound distance growth naturally — Poincaré's
distance scales logarithmically with Euclidean norm on the ball, avoiding
unbounded drift. Seed 1234's catastrophic Euclidean result is consistent
with per-seed landing in a bad scale regime.

### Components (files)

- `src/oracle_24.py` — Given `remaining`, returns winning next-ops via a
  memoized recursive search (not a tree-file lookup). Handles any state
  the model can reach, including off-tree sequences.
- `src/dagger_rollout.py` — One-problem rollout: token-by-token sampling
  with per-step z injection, step parsing, oracle labeling, invalid-step
  detection, tolerant regex for z-injection prefix artifacts.
- `src/train_stage2_dagger.py` — Two-arm Stage-2 (DAgger) trainer. `--use_z`
  flag, `--seed` override for multi-seed runs. Manual gradient averaging
  under DDP. NCCL collective timeout raised to 60 min to tolerate rank
  imbalance during variable-length rollouts.
- `configs/stage2_dagger.yaml` — Stage-2 config template.
- `scripts/run_train_stage2_dagger.sh` — Per-arm per-seed launcher:
  `bash run_train_stage2_dagger.sh <noz|z> <head_tag> [seed]`.

Artifacts:
- `checkpoints/dagger_stage2_{head_tag}/{arm_or_arm_s{seed}}/` — LoRA +
  UpProjector.
- `results/dagger_stage2_{head_tag}/{arm_or_arm_s{seed}}/{generations.jsonl, metrics.json, rollout_stats_epoch*.json}`.

See [docs/dagger_walkthrough.md](docs/dagger_walkthrough.md) for a concrete
example walkthrough (rollout terminology, oracle mechanics, z vs no-z arm
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
│   ├── train_stage2_dagger.py # stage-2 (DAgger) trainer (two-arm, DDP)
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
│   └── run_train_stage2_dagger.sh     # stage-2 per-arm per-seed
├── checkpoints/
│   ├── sft_24_tot_merged/             # frozen feature extractor
│   ├── head_{manifold}_origin_ranking/ # stage-1 heads
│   └── dagger_stage2_{head_tag}/{arm_s{seed}}/
└── results/
    ├── head_eval/{manifold}_origin_ranking/
    └── dagger_stage2_{head_tag}/{arm_s{seed}}/
```

Old v1 files (`train_plan_24.py`, `generate_24_plan.py`, `train_sft_24.py`, `train_stage1.py`, `train_stage2.py` — the teacher-forced precursor, …) remain in place as reference — not deleted so prior `results/` stay reproducible.

---

## Distributed training notes

Stage-2 (DAgger) DDP uses **manual gradient averaging** rather than `torch.nn.parallel.DistributedDataParallel`:

- Seed `torch.manual_seed(1234)` before LoRA + `UpProjector` init so every rank gets identical weights without a broadcast collective.
- After `loss.backward()`, iterate over trainable params and call `dist.all_reduce(p.grad, op=SUM) ; p.grad.div_(world_size)` before `optimizer.step()`.

Why not standard DDP? Stage-2's computation graph changes per iteration (variable-K per-boundary inner loop, plus `disable_adapter()` sub-forwards for state encoding). That makes DDP's bucket-ready ordering diverge across ranks and deadlock the first auto-reduce. Manual averaging sidesteps the problem; the sync cost is trivial for our ~22M trainable params.

NCCL topology gotcha on this host: GPUs 5↔7 are a broken pair at the NCCL level (works pair-wise with other GPUs; deadlocks when both are in the same process group). If you must use all 8 GPUs, verify with `scripts/test_nccl.sh`-style probe first.
