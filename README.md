# HypPlan: Tree-Distortion Hyperbolic Planning for LLM Reasoning

A two-stage framework. **Stage 1** teaches a small head to embed reasoning-tree states into a hyperbolic space so that hyperbolic distance reproduces tree-edge-count distance — pure geometric supervision, no language-model loss. **Stage 2** fine-tunes a new LoRA on top of a frozen SFT-merged base LLM, injecting the frozen head's geometric `z` as a virtual token before each reasoning-step boundary and optimizing next-token CE.

Runs on **Game of 24**; the MATH pipeline from the original HypPlan also lives in this tree (see `src/train_stage1.py` etc.) but the active/recommended pipeline is the two-stage Game-of-24 flow documented below.

---

## The two stages

### Stage 1 — hyperbolic head (LLM frozen)

For each Game-of-24 problem we enumerate the full state tree (root = initial 4 numbers, children = all legal (a, op, b) applications, leaves = 1-number terminal states). Each node's state text is encoded by the frozen SFT-merged Llama-3.1-8B. A small head MLP maps that hidden vector to a low-dim point (default `hyp_dim=32`) in a hyperbolic space via exp-map at the origin.

**Loss:** either MSE distortion `(d_hyp(z_i, z_j) − d_tree(i, j))²` or a Nickel-Kiela ranking loss. Both manifolds (Poincaré ball and Lorentz hyperboloid) are supported.

### Stage 2 — LoRA + up-projector (base and head frozen)

A new LoRA adapter on the SFT-merged base + a small `UpProjector` (lifts the 32-dim hyperbolic point back to hidden_dim=4096) are the only trainable parts. At each step boundary during training and inference, the canonical state-text passes through frozen base + frozen head, then up-projector, producing the virtual token `z_inj` to inject before the next step's tokens. Loss is standard next-token CE on step tokens only.

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
# Single config (defaults: poincare + distortion)
bash scripts/run_train_head.sh poincare distortion

# Full 2×2 ablation grid in parallel (one config per GPU)
bash scripts/run_stage1_grid.sh
```

Each run trains for 20 epochs on the cached hidden states (no LLM loaded), saves `checkpoints/head_{manifold}_{loss}/head.pt`, then runs `src.eval_head` to produce:

- `results/head_eval/{manifold}_{loss}/metrics.json` — mean absolute distortion, mean relative distortion, Spearman rank correlation (val + test).
- `results/head_eval/{manifold}_{loss}/scatter_{val,test}.png` — `d_tree` vs `d_hyp` scatter.
- `results/head_eval/{manifold}_{loss}/vis_tree_{idx}.png` — 2D tangent-PCA visualization of example trees.

### 3. Stage 2 — LoRA training + inference + eval

```bash
# Requires a stage-1 head first. Pass the run tag, e.g. "poincare_distortion".
bash scripts/run_train_stage2.sh poincare_distortion
```

The driver script (a) trains the LoRA + UpProjector across all detected free GPUs with DDP (manual gradient averaging — see *Distributed training notes* below), (b) generates 100 test-problem solutions with `src.generate_24_stage2`, (c) validates them with `src.evaluate_24`.

Artifacts:
- `checkpoints/hyp_stage2_{head_tag}/{lora/, up_projector.pt, config.yaml}`
- `results/hyp_stage2_{head_tag}/{generations.jsonl, metrics.json}`

### 4. Ablations

```bash
# Inference-only null: trained LoRA, random z at test time
python -m src.generate_24_stage2 \
  --stage2_checkpoint checkpoints/hyp_stage2_poincare_distortion \
  --test_data data/24_test_tot.jsonl \
  --output results/hyp_stage2_poincare_distortion/generations_randomz.jsonl \
  --random_z --limit 100

# True null baseline: re-train LoRA with random z, then eval with random z
python -m torch.distributed.run --nproc_per_node=3 -m src.train_stage2 \
  --config configs/stage2_null_randomz_train.yaml --random_z
```

`--random_z` at **training** time replaces `up_proj(head(state))` with a fresh Gaussian unit-norm 4096-vector per boundary, so the LoRA never sees any geometric signal. `--random_z` at **inference** time does the same substitution during autoregressive generation.

---

## Evaluation

`src.evaluate_24` validates each generated 3-step solution: parses `Step N: a op b = r`, replays the arithmetic, checks that all 4 input numbers are used exactly once, and confirms the final result is 24. Accuracy = fraction of problems with a valid 3-step solution.

All runs default to 100 held-out test problems from `data/24_test_tot.jsonl`. Scale via `--limit`.

---

## Results so far (100 problems, greedy decoding, ≤3 z-injections)

All Stage-3 numbers below are reported as **mean ± stdev across 3 DDP seeds
(1234, 4242, 6666)**; 2-GPU DDP is the regime used for all reported Stage-3
values because single-GPU runs showed ~3× higher seed variance (see
*Single-GPU vs DDP* below).

| System | Accuracy | Notes |
|---|---|---|
| Prior plan_24_tot (old end-to-end z training) | 0.12 | `results/24_plan_tot/` |
| Prior SFT-only baseline | 0.12 | `results/24_sft_tot/` |
| Stage-2 null (LoRA trained + tested with random z) | 0.21 | `results/hyp_stage2_null_randomz/` |
| Stage-2 best (Poincaré + distortion, teacher-forced) | 0.22 | `results/hyp_stage2_poincare_distortion/` |
| Stage-2 Poincaré + origin_ranking (teacher-forced) | 0.21 | `results/hyp_stage2_poincare_origin_ranking/` |
| **Stage-3 DAgger no-z (control, 3-seed mean)** | **0.333 ± 0.019** | `results/dagger_stage2_poincare_origin_ranking/noz_s*/` |
| **Stage-3 DAgger + Poincaré z (3-seed mean)** | **0.410 ± 0.020** | `results/dagger_stage2_poincare_origin_ranking/z_s*/` |
| **Stage-3 DAgger + Euclidean z (3-seed mean)** | 0.330 ± 0.090 | `results/dagger_stage2_euclidean_origin_ranking/z_s*/` |

**Stage-3 headline (DDP, n=3 seeds):**

| Quantity | Value |
|---|---|
| No-z → with-z Δ (Poincaré), paired | **+7.7 ± 4.2 pp** (+11, +9, +3 per seed) |
| No-z → with-z Δ (Euclidean), paired | −0.3 ± 7.6 pp (−9, +3, +5) |
| SFT → DAgger no-z (exposure-bias fix) | +21.3 pp |
| SFT → DAgger + Poincaré z (total) | +29.0 pp |

Per-seed raw numbers (DDP):

| Seed | noz | z (Poincaré) | Δ_hyp | z (Euclidean) | Δ_euc |
|---|---|---|---|---|---|
| 1234 | 0.32 | 0.43 | +11 | 0.23 | **−9** |
| 4242 | 0.32 | 0.41 | +9 | 0.35 | +3 |
| 6666 | 0.36 | 0.39 | +3 | 0.41 | +5 |

**Teacher-forced Stage-2 contributed nothing from z.** All Stage-2 z-injected
runs (0.18–0.22) were statistically indistinguishable from the null (0.21).
The ~10pp lift over SFT came from "extra LoRA fine-tuning on more data," not
from z. The fix was Stage 3 (DAgger) — see below.

**Single-GPU vs DDP.** We originally ran seed 1234 with 2-GPU DDP and three
more seeds on single-GPU. The single-GPU regime showed *much* higher
seed variance (noz = 0.18 / 0.21 / 0.35, range 17pp) because batch_size=1
makes per-step gradient noise 2× higher, and the single-GPU run does 2× as
many optimizer steps to process the same data. We therefore rely on the
3-seed DDP numbers as the headline; single-GPU numbers are kept only for
completeness under `results/dagger_stage2_poincare_origin_ranking/{noz_s2024,
z_s2024, noz_s7777, z_s7777, noz_s9999, z_s9999}/`.

Stage-1 grid (Spearman rank correlation of `d_hyp` vs `d_tree` on held-out test trees):

| manifold | loss | val Spearman | test Spearman | val abs distortion |
|---|---|---|---|---|
| lorentz | distortion | **0.772** | **0.772** | **0.291** |
| poincare | distortion | 0.763 | 0.763 | 0.315 |
| lorentz | ranking | 0.427 | 0.427 | 8.349 |
| poincare | ranking | 0.403 | 0.407 | 8.403 |

Distortion MSE beats Nickel-Kiela ranking by a wide margin on these shallow (depth ≤ 4) trees — ranking negatives sampled uniformly from all nodes don't force the head to reproduce long-range distances.

The `origin_ranking` loss (added after negative stage-2 results) explicitly
trains `|z|` to rank by solution-proximity. Target `v(s)` = BFS edge distance
from `s` to nearest success leaf in the enumerated tree. For any sampled pair
`(s_i, s_j)` with `v(s_i) < v(s_j)`, hinge loss
`max(0, d_H(z_i, 0) − d_H(z_j, 0) + margin)`. This head is the one used
downstream by Stage 3 (DAgger).

---

## Stage 3 — DAgger with tree oracle

### Diagnosis of stage-2 failure

Stage 2 as implemented (CE-only training with teacher-forced trajectories)
fails to extract signal from z for two compounding reasons:

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

### Fix: expert-iteration framing (AlphaGo-style)

Treat the head as a **privileged critic** (stage-1 training had access to the
full enumerated tree — solution locations, distance to nearest success —
information the policy would otherwise have to re-derive). Treat the LoRA as
a **policy** trained under the distribution of states it actually reaches at
inference, using oracle-labeled targets.

The key move is to abandon teacher forcing. Under free generation the model
can reach genuinely uncertain states; z then carries decision-relevant
information the model cannot trivially recompute from context.

### DAgger training loop

At each epoch, for each training problem `P`:

1. **Rollout under current policy.** Generate step-by-step from the prompt,
   with temperature `T=0.7` and top-p `0.95` at each step. Inject `z_t` as a
   virtual token at each step boundary (z-arm only; no-z arm omits the
   injection). Continue until either (a) model emits a valid 3-step solution,
   (b) model emits an invalid step, or (c) a step budget is exhausted.
2. **Oracle labeling.** For each step-boundary state `s_t` reached, query the
   oracle: given the current `remaining` multiset, return the set of winning
   next ops (ops `(a, op, b)` whose resulting state can still reach 24 in the
   remaining step budget).
3. **Invalid-step handling.** If a step is unparseable, does arithmetic wrong,
   or uses numbers not in `remaining`, the trajectory is **truncated at that
   step**. Earlier valid states still contribute to training. Track
   invalid-step rate as a primary metric; alarm if >50% post-warm-start.
4. **Training pass.** For each collected `(s_t, z_t, winning_ops_t)` tuple,
   pick one winner deterministically (lex-order on `(op, a, b)`) and CE-train
   the model to emit its full step text. Backprop into LoRA + UpProjector;
   head and base remain frozen.
   - *Phase-1 loss: single-winner CE.* All winners at a given step share
     `v(s') = len(s')−1` on a shallow 3-step tree, so v-based tiebreaking
     doesn't differentiate — lex is as principled as anything else and keeps
     compute to one forward pass per training example.
   - *Phase-2 upgrade path:* log-of-sum over full step texts of all winners
     via gradient-accumulated softmax reweighting (≈K× compute). Not needed
     to produce the current results.

### Two-arm experimental design

Both arms use **identical** code path, warm start, sampling hyperparams,
oracle rules, and DAgger schedule. A single `--use_z` flag toggles z-injection
on/off. This isolates z's contribution from the exposure-bias fix, both of
which independently should raise accuracy. The clean metric:
`Δ_accuracy = acc(z-arm) − acc(no-z-arm)`.

### Warm start (critical design choice)

**Do NOT warm-start from existing stage-2 checkpoints.** Our null-baseline
evidence shows those LoRAs learned z's distribution as a calibration signal,
not its content. Starting from them would inherit attention patterns that
route *around* z's semantic content — exactly the local minimum DAgger needs
to escape.

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

### Results (3-seed DDP)

See the headline table at the top. Summary:

- **DAgger alone (no z)**: 0.333 ± 0.019 — +21pp over the 0.12 SFT baseline,
  +12pp over the prior stage-2 null (0.21). This is the exposure-bias fix:
  training on model-reached states teaches recovery behavior that teacher
  forcing can't.
- **DAgger + Poincaré z**: 0.410 ± 0.020 — adds **+7.7 ± 4.2pp** on top,
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
- `src/train_stage2_dagger.py` — Two-arm DAgger trainer. `--use_z` flag,
  `--seed` override for multi-seed runs. Manual gradient averaging under
  DDP. NCCL collective timeout raised to 60 min to tolerate rank
  imbalance during variable-length rollouts.
- `configs/stage2_dagger.yaml` — DAgger config template.
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
│   ├── head.yaml              # stage-1 template (manifold + loss switchable)
│   ├── stage2.yaml            # teacher-forced stage-2 template (legacy)
│   └── stage2_dagger.yaml     # DAgger stage-3 template
├── src/
│   ├── tree_data.py           # enumerate_tree, render_state, pair_distances_lca
│   ├── hyperbolic.py          # Lorentz ops (unchanged from v1)
│   ├── head.py                # HyperbolicHead (Poincaré/Lorentz/Euclidean) + UpProjector
│   ├── train_head.py          # stage-1 trainer (distortion/ranking/origin_ranking)
│   ├── eval_head.py           # distortion/Spearman + 2D viz
│   ├── oracle_24.py           # stage-3 oracle: winning_ops(remaining)
│   ├── dagger_rollout.py      # stage-3 rollout + oracle labeling
│   ├── dataset_24_stage2.py   # per-boundary canonical state tokenization
│   ├── train_stage2.py        # teacher-forced stage-2 trainer (legacy)
│   ├── train_stage2_dagger.py # DAgger stage-3 trainer (two-arm, DDP)
│   ├── generate_24_stage2.py  # inference (supports --no_z_inject, --random_z)
│   └── evaluate_24.py         # solution validator (unchanged)
├── data/
│   ├── generate_tree_data.py  # offline tree + hidden-state cache builder
│   ├── 24_{train,val,test}.jsonl
│   └── trees/                 # cached tree metadata + hidden states
├── docs/
│   └── dagger_walkthrough.md  # concrete example of DAgger mechanics
├── scripts/
│   ├── run_gen_tree_data.sh
│   ├── run_train_head.sh
│   ├── run_stage1_grid.sh
│   ├── run_train_stage2.sh            # legacy teacher-forced stage-2
│   └── run_train_stage2_dagger.sh     # DAgger stage-3 per-arm per-seed
├── checkpoints/
│   ├── sft_24_tot_merged/             # frozen feature extractor
│   ├── head_{manifold}_{loss}/        # stage-1 heads
│   └── dagger_stage2_{head_tag}/{arm_s{seed}}/
└── results/
    ├── head_eval/{manifold}_{loss}/
    ├── hyp_stage2_{head_tag}/         # teacher-forced stage-2
    └── dagger_stage2_{head_tag}/{arm_s{seed}}/
```

Old v1 files (`train_plan_24.py`, `generate_24_plan.py`, `train_sft_24.py`, `train_stage1.py`, …) remain in place as reference — not deleted so prior `results/` stay reproducible.

---

## Distributed training notes

Stage-2 DDP uses **manual gradient averaging** rather than `torch.nn.parallel.DistributedDataParallel`:

- Seed `torch.manual_seed(1234)` before LoRA + `UpProjector` init so every rank gets identical weights without a broadcast collective.
- After `loss.backward()`, iterate over trainable params and call `dist.all_reduce(p.grad, op=SUM) ; p.grad.div_(world_size)` before `optimizer.step()`.

Why not standard DDP? Stage 2's computation graph changes per iteration (variable-K per-boundary inner loop, plus `disable_adapter()` sub-forwards for state encoding). That makes DDP's bucket-ready ordering diverge across ranks and deadlock the first auto-reduce. Manual averaging sidesteps the problem; the sync cost is trivial for our ~22M trainable params.

NCCL topology gotcha on this host: GPUs 5↔7 are a broken pair at the NCCL level (works pair-wise with other GPUs; deadlocks when both are in the same process group). If you must use all 8 GPUs, verify with `scripts/test_nccl.sh`-style probe first.
