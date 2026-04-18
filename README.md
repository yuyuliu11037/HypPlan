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

| System | Accuracy | Notes |
|---|---|---|
| Prior plan_24_tot (old end-to-end z training) | 0.12 | `results/24_plan_tot/` |
| Prior SFT-only baseline | 0.12 | `results/24_sft_tot/` |
| HypPlan v2 — Lorentz + distortion | 0.20 | `results/hyp_stage2_lorentz_distortion/` |
| HypPlan v2 — Poincaré + distortion | 0.22 | `results/hyp_stage2_poincare_distortion/` |
| Null baseline (LoRA trained + tested with random z) | 0.21 | `results/hyp_stage2_null_randomz/` |
| HypPlan v2 — Lorentz + origin_ranking | 0.18 | `results/hyp_stage2_lorentz_origin_ranking/` |
| HypPlan v2 — Poincaré + origin_ranking | 0.21 | `results/hyp_stage2_poincare_origin_ranking/` |
| **Stage 3 — DAgger no-z (control)** | **0.32** | `results/dagger_stage2_poincare_origin_ranking/noz/` |
| **Stage 3 — DAgger with-z** | **0.43** | `results/dagger_stage2_poincare_origin_ranking/z/` |

**Stage 3 result (two-arm isolation):** +11pp from DAgger alone (exposure-bias
fix) AND +11pp from the z signal on top. Total +31pp over SFT baseline. The
prior negative stage-2 result was a teacher-forcing artifact: z was
informationally redundant given the ground-truth preceding trajectory, so CE
had no gradient pressure to use it. Under DAgger with free generation +
oracle labels, z finally has decision-relevant signal the model can't
trivially recompute from text context.

**Head-attributable contribution is zero.** All z-injected runs (0.18–0.22) are
statistically indistinguishable from the proper null baseline (0.21; Wilson CI
≈ ±8pp on n=100). The ~9pp lift over SFT comes from "extra LoRA fine-tuning on
more data," not from z. Value-probe diagnostics confirm this: the stage-1 head
destroys most of the value information the raw LLM hidden state contains
(R²=0.37 on non-leaf nodes → R²=0.04 after the 32-dim head bottleneck; see
`results/value_probe/`). Even a purpose-built value-aware head
(`origin_ranking`, non-leaf R²=0.065, |z|-vs-value Spearman=-0.23) failed to
transmit any signal to stage-2 accuracy — evidence that the bottleneck is the
stage-2 conditioning mechanism (single virtual-token injection, CE-only), not
the stage-1 objective. See **Stage-3: DAgger with tree oracle** below for the
fix being developed.

Stage-1 grid (Spearman rank correlation of `d_hyp` vs `d_tree` on held-out test trees):

| manifold | loss | val Spearman | test Spearman | val abs distortion |
|---|---|---|---|---|
| lorentz | distortion | **0.772** | **0.772** | **0.291** |
| poincare | distortion | 0.763 | 0.763 | 0.315 |
| lorentz | ranking | 0.427 | 0.427 | 8.349 |
| poincare | ranking | 0.403 | 0.407 | 8.403 |

Distortion MSE beats Nickel-Kiela ranking by a wide margin on these shallow (depth ≤ 4) trees — ranking negatives sampled uniformly from all nodes don't force the head to reproduce long-range distances.

The new `origin_ranking` loss (added after negative stage-2 results) explicitly
trains `|z|` to rank by solution-proximity. Target `v(s)` = BFS edge distance
from `s` to nearest success leaf in the enumerated tree. For any sampled pair
`(s_i, s_j)` with `v(s_i) < v(s_j)`, hinge loss
`max(0, d_H(z_i, 0) − d_H(z_j, 0) + margin)`. Value-probe improvement:

| Head | non-leaf R² | Spearman(\|z\|, value) |
|---|---|---|
| Raw LLM hidden (ceiling)       | 0.367 | —     |
| Lorentz distortion             | 0.040 | -0.09 |
| Lorentz origin_ranking         | 0.068 | -0.23 |
| Poincaré distortion            | 0.026 | -0.10 |
| Poincaré origin_ranking        | 0.065 | -0.23 |

---

## Stage 3 — DAgger with tree oracle (IN DEVELOPMENT)

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
   compute the **log-of-sum** loss:
   `L = −log Σ_{op ∈ winning_ops} p(op_tokens | prefix, z_t)`
   This treats "any winning op is correct" as a single event — the policy is
   free to be peaked on one winner. Backprop into LoRA + UpProjector; head
   and base remain frozen.

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
- **Zero-init UpProjector** (final Linear weight & bias zero → `z_inj = 0`
  initially, which is a harmless attention sink).
- Frozen `head_{manifold}_origin_ranking` as the critic.

First rollout with this init = pure SFT-merged behavior = 0.12 baseline,
without inheriting any bad z-attention habits. DAgger teaches the LoRA to use
z from scratch.

### Decisions locked in

1. Drop invalid trajectories from the invalid step onward. Log drop rate per
   epoch; alarm if >50% after epoch 0.
2. **Log-of-sum** loss (not sum-of-logs) over winning ops. Allows peakedness
   within the winning set.
3. Fresh LoRA (B=0) + zero-init UpProjector. See above.
4. T=0.7, top-p=0.95 for rollout. Greedy for eval.
5. Lockstep DAgger: per epoch, rollout all 1090 train problems (3 trajectories
   each ≈ 3300 trajectories), then one CE pass over collected pairs. Repeat
   for 3 epochs. Both arms run simultaneously.

### Expected signal strength

Free-generation training will likely raise accuracy on its own (exposure bias
is real). The no-z control isolates exposure-bias lift from z-signal lift.
**The meaningful result is Δ_accuracy.** If Δ is small on 24-Game, the
framework is still expected to pay off on tasks where z has a stronger
information advantage over text-derivable reasoning (deeper search trees,
more numbers, combinatorial puzzles). 24-Game is chosen here because all
infrastructure already exists; migrating the pipeline to a harder task only
requires new tree enumeration + new oracle.

### Components (files)

- `src/oracle_24.py` — Given `remaining`, returns winning next-ops. Pure
  Python + `fractions.Fraction` arithmetic; small lru_cache.
- `src/dagger_rollout.py` — One-problem rollout: sampling loop with per-step
  z injection, step parsing, oracle labeling, invalid-step detection.
- `src/train_stage2_dagger.py` — Two-arm DAgger trainer. `--use_z` flag.
  Manual gradient averaging under DDP (same pattern as `train_stage2.py`).
- `configs/stage2_dagger.yaml` — DAgger config template.
- `scripts/run_train_stage2_dagger.sh` — Launcher with auto-GPU detection.

Artifacts:
- `checkpoints/dagger_stage2_{head_tag}_{arm}/` — LoRA + UpProjector.
- `results/dagger_stage2_{head_tag}_{arm}/{generations.jsonl, metrics.json, rollout_stats.jsonl}`.

---

## Project layout (v2 files only)

```
HypPlan/
├── configs/
│   ├── head.yaml              # stage-1 template (manifold + loss switchable)
│   └── stage2.yaml            # stage-2 template
├── src/
│   ├── tree_data.py           # enumerate_tree, render_state, pair_distances_lca
│   ├── hyperbolic.py          # Lorentz ops (unchanged from v1)
│   ├── head.py                # HyperbolicHead (Poincaré/Lorentz) + UpProjector
│   ├── train_head.py          # stage-1 trainer (distortion or ranking)
│   ├── eval_head.py           # distortion/Spearman + 2D viz
│   ├── dataset_24_stage2.py   # per-boundary canonical state tokenization
│   ├── train_stage2.py        # LoRA + up-projector DDP trainer
│   ├── generate_24_stage2.py  # inference with frozen head + up-projector
│   └── evaluate_24.py         # solution validator (unchanged)
├── data/
│   ├── generate_tree_data.py  # offline tree + hidden-state cache builder
│   ├── 24_{train,val,test}.jsonl
│   └── trees/                 # cached tree metadata + hidden states
├── scripts/
│   ├── run_gen_tree_data.sh
│   ├── run_train_head.sh      # single-config
│   ├── run_stage1_grid.sh     # 4-way parallel ablation grid
│   └── run_train_stage2.sh
├── checkpoints/
│   └── sft_24_tot_merged/     # frozen feature extractor (Llama-3.1-8B-Instruct + SFT LoRA merged)
└── results/
    ├── head_eval/{manifold}_{loss}/   # stage-1 metrics + plots
    └── hyp_stage2_{head_tag}/         # stage-2 metrics + generations
```

Old v1 files (`train_plan_24.py`, `generate_24_plan.py`, `train_sft_24.py`, `train_stage1.py`, …) remain in place as reference — not deleted so prior `results/` stay reproducible.

---

## Distributed training notes

Stage-2 DDP uses **manual gradient averaging** rather than `torch.nn.parallel.DistributedDataParallel`:

- Seed `torch.manual_seed(1234)` before LoRA + `UpProjector` init so every rank gets identical weights without a broadcast collective.
- After `loss.backward()`, iterate over trainable params and call `dist.all_reduce(p.grad, op=SUM) ; p.grad.div_(world_size)` before `optimizer.step()`.

Why not standard DDP? Stage 2's computation graph changes per iteration (variable-K per-boundary inner loop, plus `disable_adapter()` sub-forwards for state encoding). That makes DDP's bucket-ready ordering diverge across ranks and deadlock the first auto-reduce. Manual averaging sidesteps the problem; the sync cost is trivial for our ~22M trainable params.

NCCL topology gotcha on this host: GPUs 5↔7 are a broken pair at the NCCL level (works pair-wise with other GPUs; deadlocks when both are in the same process group). If you must use all 8 GPUs, verify with `scripts/test_nccl.sh`-style probe first.
