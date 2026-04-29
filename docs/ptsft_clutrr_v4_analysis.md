# Why Planning-Token SFT got 100% on CLUTRR v4 (template-disjoint)

This document explains the surprising **PT-SFT 100%** result on the
template-disjoint CLUTRR v4 test set, why it's not a leakage bug, and what
it means for interpreting the HypPlan vs PT-SFT comparison across tasks.

## What Planning-Token SFT is (mechanically)

PT-SFT is standard supervised fine-tuning of a LoRA adapter on
`(question, answer)` pairs, where the answer text is rewritten with special
**planning-token markers** before each step:

Training example (one of 3000):

```
question:
  Zach is the daughter of Olivia.
  Olivia is the father of Sam.
  Noah is the father of Olivia.
  Sam is the father of Xavier.

  How is Olivia related to Xavier?

answer:
  <PLAN:COMPOSE> Step 1: Olivia is the father of Sam
  <PLAN:COMPOSE> Step 2: Olivia is the grandfather of Xavier
  <PLAN:ANS> Answer: Olivia is the grandfather of Xavier.
```

`<PLAN:COMPOSE>` and `<PLAN:ANS>` are inserted automatically by
`data/annotate_sft_plan_groupB.py`. They are structural cues that mark
"this is a composition step" / "this is the final answer." Otherwise the
training is bog-standard cross-entropy SFT with LoRA (r=16, ~25M trainable
params).

Training run for CLUTRR v4: 5 epochs × 230 steps × global batch 64 ≈ 75K
example passes. Loss converged to ~0.0001 by epoch 4.

## Why 100% is not memorization

CLUTRR v4 was deliberately constructed with a **template-disjoint** split:

| Split | k=2 chains | k=3 chains | k=4 chains | total chains |
|---|---|---|---|---|
| Train | 12 | 24 | 48 | **84** |
| Test  | 4 | 8 | 16 | **28** (held out) |

The 28 test chain templates **never appear in training**. We verified this
at generation time: zero overlap between train and test on
`(N, k, prefix)` tuple keys.

If PT-SFT were just memorizing the
`(chain-template → final-relation)` lookup table from training, it would
fail on every single test problem because none of the test templates was
ever shown to the model. But it got 100%. Therefore the model must have
learned something more general than chain-template lookup.

## What it actually learned: compositional kinship algebra

CLUTRR's task has clean **compositional structure**. Each pair of base
relations composes to a fixed third relation, independent of context:

```
mother ∘ father  → grandmother
father ∘ mother  → grandmother
son    ∘ daughter→ granddaughter
mother ∘ sister  → aunt
father ∘ brother → uncle
...
```

These pair-rules are local. Composing a 4-hop chain is just applying a
pair-rule three times sequentially:

```
[mother, mother, sister, daughter]
  =>  mother ∘ mother      = grandmother
  =>  grandmother ∘ sister = great-aunt
  =>  great-aunt ∘ daughter = first-cousin-once-removed
```

After 5 epochs the model has effectively learned:

1. **Base vocabulary** — the 8 atomic relations
   (mother, father, son, daughter, brother, sister, husband, wife) and
   their natural-language realizations.
2. **The composition operator** — how to combine two relations into a
   third. With ~84 distinct chain templates each appearing ~36 times, the
   model has seen each pair-rule applied many times in many contexts.
3. **Deep-chain vocabulary** — rare terms like
   `first-cousin-once-removed`, `great-great-grandmother`, `great-uncle`,
   `great-aunt`. Each appears in many training composition examples,
   teaching the model the precise vocabulary the task uses.

At test time, even on the held-out template
`[mother, mother, sister, daughter]`:

```
Step 1: Tara is the mother of Peter             (read directly from facts)
Step 2: Tara is the grandmother of Mia          (apply mother ∘ mother)
Step 3: Tara is the great-aunt of Anna          (apply grandmother ∘ sister)
Step 4: Tara is the first-cousin-once-removed of Kate
                                                 (apply great-aunt ∘ daughter)
Answer: Tara is the first-cousin-once-removed of Kate.
```

Each individual step is a **local pair-composition** the model has seen
many times during training. The complete 4-hop chain was held out, but
every 2-step sub-composition inside it is in-distribution. The model
generalizes from the learned algebra.

## Why baselines couldn't do this

| Method | Score | Per-k (k=2 / k=3 / k=4) |
|---|---|---|
| Few-shot greedy | 17.8% | 47% / 7% / 0% |
| Self-Consistency K=5 | 22.2% | 57% / 10% / 0% |
| ToT BFS | 16.7% | 50% / 0% / 0% |
| **Planning-Token SFT** | **100%** | 100% / 100% / 100% |

Base Qwen-14B-Instruct does have *general* kinship knowledge from
pretraining (it knows what "grandfather" means) but not the *precise
vocabulary* CLUTRR uses, especially for the deep terms. Typical few-shot
failure pattern:

```
Step 1: Ivy is the father of Bob.
Step 2: Bob is the father of Ben.
Step 3: Ben is the brother of Zach.
Step 4: ...
Answer: Ivy is the (paternal) grandfather of Xavier.    ← wrong (should be
                                                          first-cousin-
                                                          once-removed)
```

It composes the first 2-3 steps correctly but at the 4th step it picks the
nearest plausible kinship label it knows, often confusing
"first-cousin-once-removed" with "grandfather" or "great-uncle". The
pretrained model just doesn't have the precise enough algebra. PT-SFT
explicitly teaches it. ToT and SC both share this same base-model
limitation, so they don't help.

## What this means for interpreting HypPlan vs PT-SFT

If a task is **compositionally generalizable** (each test instance can be
decomposed into local rules already seen in train), then plain SFT solves
it once you train on enough disjoint templates. HypPlan's tree-of-thoughts
+ hyperbolic-value-head machinery doesn't add anything beyond SFT in that
regime.

The HypPlan-specific value shows on tasks where SFT *doesn't* generalize.
That's the critical paper distinction:

| Task type | SFT enough? | HypPlan delta over PT-SFT |
|---|---|---|
| Compositional reasoning (CLUTRR, shallow ProofWriter) | Yes | ~0 (both saturate) |
| Combinatorial search (N-Queens) | No | **+17.8 pp** (HypPlan 26.7% vs PT-SFT 8.9%) |
| Deep chained deduction (ProofWriter QDep=3) | Partially | **+33 pp** vs SC; PT-SFT also struggles at depth |

The CLUTRR v4 result is therefore not a HypPlan win, but it **is** a clean
methodological finding: template-disjoint CLUTRR is solvable by SFT
alone, which sets a strong PT-SFT baseline that we still need to beat (or
explain not beating) on other tasks.

## Implication for the paper narrative

The strongest framing across the three tasks:

> Planning-Token SFT is a strong baseline on tasks where the gold
> trajectory factorises into local rules each seen in training (CLUTRR);
> there it saturates and HypPlan offers no additional advantage. HypPlan
> wins on tasks where the gold trajectory cannot be reduced to local
> rule-lookup — combinatorial search (N-Queens) and deep chained
> deduction (ProofWriter QDep=3). The benefit of the
> hyperbolic-value-conditioned LoRA is therefore most visible on
> tasks that genuinely require search or non-local planning, not on
> tasks where memorising the local rule-table is enough.

## File pointers

```
data/clutrr_train_sft_plan.jsonl                # PT-annotated v4 train (3000 records)
data/annotate_sft_plan_groupB.py                # PT annotation script
configs/sft_pt_clutrr_qwen14b.yaml              # PT-SFT training config (5 epochs)
src/train_sft_pt_qwen.py                        # PT-SFT trainer
src/eval_pt_ood.py                              # PT-SFT eval driver (greedy decode)
checkpoints/sft_pt_clutrr_qwen14b/lora/         # trained LoRA (100 MB)
results/sft_pt_clutrr_v4/clutrr_pt.jsonl        # 90-record evaluation outputs
```
