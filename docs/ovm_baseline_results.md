# OVM (Outcome-supervised Value Models) baseline reproduction

We reproduce OVM (Yu et al., NAACL Findings 2024,
[arxiv.org/abs/2311.09724](https://arxiv.org/abs/2311.09724)) as a
baseline for HypPlan. OVM is the natural Euclidean-counterpart to
HypPlan: both train a value model from outcome supervision; OVM uses
a single scalar head and applies it via post-hoc step-level beam search,
HypPlan uses a hyperbolic head whose output is injected into the LoRA
during generation.

If HypPlan ≈ OVM, geometry doesn't help — the gain is from "any trained
value head." If HypPlan > OVM, the hyperbolic geometry / per-state
z-injection mechanism matters.

## Method (concise)

| Component | Implementation |
|---|---|
| Generator | Qwen-14B-Instruct + per-task PT-SFT LoRA (re-used) |
| Rollout sampling | Temperature 1.0, top-p 0.95, 20 rollouts/problem (10K trajectories for G24; ~5–10K for the others) |
| Outcome label | 1 if rollout's final answer matches gold via the task's existing scorer, else 0 |
| Value head | `nn.Linear(hidden_dim, 1)` → sigmoid; base + LoRA frozen |
| Loss | Per-token MSE on assistant tokens; `pos_weight` for imbalanced tasks |
| Step boundary | Newline character |
| Inference (step-beam) | K=4 candidates per beam, beam=3, max_steps=10–15 (K=20 b=5 caused OOM at 14B; K=4 b=3 fits comfortably) |

## Pipeline files

| File | Purpose |
|---|---|
| [src/ovm_head.py](../src/ovm_head.py) | scalar value head |
| [scripts/gen_ovm_rollouts.py](../scripts/gen_ovm_rollouts.py) | rollout generator with heartbeats, sharded |
| [src/train_ovm.py](../src/train_ovm.py) | DDP gloo trainer with class-imbalance `pos_weight` and phantom-zero-loss safeguard |
| [src/eval_ovm.py](../src/eval_ovm.py) | step-beam inference with batched value scoring |
| `configs/ovm_{g24,nqueens,bw,gc}_qwen14b.yaml` | per-task configs |

## Results

All numbers on the same per-task test sets used elsewhere in this
repo. Final-answer accuracy.

### Full-test-set numbers

| Task | Test size | SC K=5 | **OVM (ours)** | HypPlan in-domain |
|---|---|---|---|---|
| **G24** | 100 | 21/100 = 21% | **3/100 = 3%** | (not run) |
| **N-Queens** | 45 | 5/45 = 11.1% | **2/45 = 4.4%** | 12/45 = 26.7% |
| **Blocksworld** | 200 | 119/200 = 60% | **163/200 = 81.5%** | 67/100 = 67% (subset) |
| **Graphcolor** | 200 | 132/200 = 66% | **119/200 = 59.5%** | 88/100 = 88% (subset) |

### Direct comparison on identical 100-record HypPlan subsets

For BW and GC, the prior HypPlan eval used a 100-record subset of the
respective 200-record test files. Restricting OVM to those same 100
records gives a clean head-to-head:

| Task | Subset size | OVM (this work) | HypPlan in-domain | Δ |
|---|---|---|---|---|
| Blocksworld | 100 (HypPlan subset) | 79% | 67% | **OVM +12pp** |
| Graphcolor | 100 (HypPlan subset) | 58% | 88% | **HypPlan +30pp** |
| N-Queens | 45 (full) | 4.4% | 26.7% | **HypPlan +22pp** |

### When OVM works

**Blocksworld (+21.5pp over SC):** generator p_correct=82% at T=1.0
gives the value head ample positive trajectories. Step-beam at K=4
b=3 reliably selects a correct plan from 4·3=12 candidates per step,
each ≈82% likely correct individually, so the beam almost always
contains a correct trajectory; the value head picks the right one.

This is the OVM-paper regime: a strong generator + outcome-labeled
trajectories + step-beam selection.

### When OVM fails

**N-Queens (4.4% vs PT-SFT 8.9%):** N-Queens correctness depends on
*non-local* board constraints (no two queens may attack across the full
board). Token-level value supervision can't reliably predict
final-board validity from a partial placement, because the constraint
violation only manifests when subsequent placements fail. The trained
head essentially predicts mean outcome; step-beam degenerates to
random sampling at T=1.0.

**G24 (3% matches PT-SFT 3%):** the PT-SFT generator overfits to the
training distribution and only achieves ~3% greedy on the
`24_test.jsonl` test split. With only ~6% positive rollouts at T=1.0,
the value head trained even with `pos_weight=15` produces poorly
calibrated scores. OVM is generator-limited: cannot amplify a signal
that isn't there.

**Graphcolor (-6.5pp vs SC):** generator p_correct=52% at T=1.0 is
healthy, but step-beam's K=4 b=3 doesn't add enough value over SC's
K=5 majority vote, which already has a strong inductive prior
(majority over independent samples). OVM's compute budget is similar
to SC but SC's vote is a stronger aggregator for short, format-rigid
outputs.

## Key engineering pitfalls (documented for future replication)

1. **Class imbalance kills MSE.** With p_correct ≪ 50%, the value
   head trivially learns to predict ~0 everywhere. Fix: add
   `pos_weight = 1/p_correct` to scale positive-label loss.
   ([train_ovm.py:227](../src/train_ovm.py#L227))
2. **Temperature must match the generator's calibration.** BW PT-SFT
   produces 93% correct at T=0.1 but 0% at T=1.0 (formatting drifts
   destroy plans). Choose T such that p_correct lands in 30–80%.
3. **Two scoring bugs cost ~2 hours each.** (a) `_score("bw", ...)`
   used `rec.get("prompt", "")` but TRAIN data has only `question` —
   silent zero scoring. (b) `eval_ovm` dropped `prompt` from the
   output JSONL but the BW scorer needs it for goal-extraction —
   silent zero scoring. Both fixed.
4. **K=20 OOMs on 14B with 400-token prompts.** Reduce K to 4 + use
   batched value scoring; total per-problem forward count drops from
   1000+ to ~100.

## Implication for the HypPlan paper narrative

OVM is a "trained-value-head" baseline that demonstrates the value of
outcome supervision *without* hyperbolic geometry. The 3-task direct
comparison (same test subsets) gives:

| Task | OVM | HypPlan | Winner |
|---|---|---|---|
| N-Queens | 4.4% | 26.7% | **HypPlan +22pp** |
| Blocksworld | 79% | 67% | **OVM +12pp** |
| Graphcolor | 58% | 88% | **HypPlan +30pp** |

The two methods solve **different failure modes**:

- **HypPlan dominates on constraint-satisfaction tasks** (N-Queens,
  Graphcolor): correctness depends on *non-local* properties of the
  full trajectory (no two queens attack; no two adjacent vertices
  share a color). OVM's token-level scalar value can't reliably
  predict these — partial trajectories look fine until a constraint
  is violated downstream. HypPlan's per-state z-injection conditions
  the LoRA on the full input graph at *every* step, so the policy
  itself avoids constraint violations rather than filtering them
  post-hoc.
- **OVM dominates on plan-execution tasks** (Blocksworld): each step
  is a single legal action; correctness compounds linearly along the
  plan. The token-level outcome supervision captures this exactly.
  Step-beam search at K=4 b=3 with a strong generator (82% T=1.0
  success) reliably finds and selects a goal-achieving plan.

This split is a **complementary-strengths** story rather than a
HypPlan-strictly-better one. The paper claim becomes: HypPlan's
hyperbolic z-conditioning is the right inductive bias for tasks where
local actions are valid but global correctness depends on the full
trajectory; OVM is the right baseline-of-trained-value-heads to
demonstrate this distinction.
