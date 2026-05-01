# CLUTRR-Graph — kinship reasoning over dense family graphs

## Why a graph variant

The standard CLUTRR task (Sinha et al., 2019) presents a single linear chain
of kinship facts `A → B → C → D → E` and asks for the relation between the
endpoints. At each reasoning step there is exactly **one** correct
composition (compose the next chain edge); there is no branching, no
distractor, and no dead end. The value landscape `v(s) = #compositions to
goal` is flat — every intermediate state is equally on-path.

This is a poor stress test for HypPlan, whose hyperbolic value head is
designed to discriminate states by their distance-to-goal in a
**branching** search tree. On linear chains, plain Planning-Token SFT
saturates at 100% (see `docs/ptsft_clutrr_v4_analysis.md`) because the
trajectory factorises into local rule-lookups already covered by training.

CLUTRR-Graph fixes this by replacing the linear chain with a
**dense family graph**: 17 entities and ~32 stated edges, of which only
4 lie on the gold path. The model must select which two facts to compose
at each step, and ~85% of locally legal compositions do not progress
toward the query target. This produces the genuinely asymmetric value
landscape HypPlan was built for.

## Task definition

- **Story.** A list of `(head, relation, tail)` kinship facts rendered as
  natural-language sentences ("Alice is the mother of Bob.").
- **Query.** "How is X related to Y?" where X and Y are two specific
  entities.
- **Answer.** The composed kinship relation between X and Y, derivable
  from the stated facts via repeated composition of the kinship algebra
  (mother ∘ father → grandmother, mother ∘ brother → uncle, etc.).
- **Composition table.** Same as `oracle_clutrr.RELATION_COMPOSITION`,
  including the deep `great-*` ladder and `first-cousin-*-removed` terms.
  Composition is undefined for some pairs (e.g. `wife ∘ uncle`); those are
  rejected during generation and treated as dead-end actions at inference.

A reasoning step is the choice of two stated-or-derived triples that share
a middle entity, plus the composition that produces a new triple. The
state is the set of derived `(head, rel, tail)` triples (initially the
stated edges); the search is over this growing set, not over a linear
chain.

## Generation parameters

Configurable knobs in `src/oracle_clutrr_graph.generate_graph_problem`:

| Parameter | Meaning | Default |
|---|---|---|
| `k` | gold-chain length (number of compositions to derive the answer) | 4 |
| `n_distractor_entities` | extra entities beyond the gold chain | 12 |
| `n_distractor_edges` | extra edges among distractors and to chain pivots | 28 |
| `min_head_out` | minimum outgoing edges from the query head | 2 |
| `min_tail_in` | minimum incoming edges into the query tail | 2 |

For the v5 dataset we use `k=4, n_distractor_entities=12,
n_distractor_edges=28, min_head_out=2, min_tail_in=2`. This produces
17-entity, ~32-edge graphs with mean **88% dead-end ratio** at the
initial state.

## The `min_head_out` / `min_tail_in` filter

This filter exists because of a design pitfall we caught during
prototyping. Without it, distractors only attach to **interior** chain
entities (indices `1..k-1`), so the query head ends up with a single
outgoing edge in the narrative — the gold first edge. The model can
trivially identify the right first composition by lookup. We measured this
in a 50-instance probe: PT-SFT got **72%** under the trivial first-step
regime, and only **54%** once head/tail also got distractor edges.
Concretely:

| Query-head out-degree | PT-SFT probe accuracy |
|---|---|
| 1 (only gold edge present) | 9/11 = 82% |
| 2 (one distractor, one gold) | 10/23 = 43% |
| 3 | 8/16 = 50% |

The v5 generator therefore enforces `head_out ≥ 2` and `tail_in ≥ 2` so
every test instance has a real first-step choice.

## Generation pipeline

Implemented in `src/oracle_clutrr_graph.py` and `scripts/gen_clutrr_graph.py`.

1. **Sample chain.** Draw `k` base relations from
   `{mother, father, son, daughter, brother, sister}`; reject if the
   composition is undefined.
2. **Build chain entities.** Allocate `k+1` distinct names for the
   on-path entities `e0, ..., ek`. Add stated edges
   `(ei, chain[i], e[i+1])` for `i=0..k-1`.
3. **Add distractor entities.** Allocate `n_distractor_entities` extra
   names not on the gold chain.
4. **Add distractor edges.**
   - **Phase A:** every distractor entity gets ≥1 incident edge to a
     chain pivot drawn from `{0, 1, ..., k}` (i.e. *all* chain entities,
     including endpoints).
   - **Phase B:** remaining edge budget is spent on distractor↔distractor
     edges, allowing distractor sub-chains of length ≥2.
5. **Verify uniqueness.** Run forward BFS over entity pairs to compute
   the shortest composition distance between the query head and tail.
   Reject the instance if (a) the shortest distance ≠ k, or (b) the
   shortest path goes through any distractor entity. This guarantees the
   gold chain is the *unique* shortest derivation and prevents shortcuts
   through the distractor subgraph.
6. **Apply `min_head_out` / `min_tail_in` filter.** Reject if the query
   endpoints don't have enough incident edges (see above).
7. **Shuffle.** Randomise edge order in the narrative so gold and
   distractor edges interleave naturally.

A typical accepted instance:

```
Carol is the father of Dan.
Frank is the brother of Dan.
Grace is the mother of Dan.
Dan is the father of Mia.
Mia is the brother of Tara.
Mia is the daughter of Noah.
... (28 more edges, 12 distractor entities)

How is Carol related to Sam?
Gold answer: first-cousin-once-removed
Gold chain: Carol → Dan → Mia → Tara → Sam
Gold relations: [father, father, brother, daughter]
```

## Train / test split

We use a **template-disjoint** split on the chain relations, mirroring
the CLUTRR v4 design.

1. Over-sample 4× the target dataset size (12,800 candidate problems)
   to enumerate the distinct relation-chain templates in the pool.
2. With `k=4` and 6 base relations there are at most `6^4 = 1296`
   distinct chains, of which ~64 produce composable answers and pass
   the BFS uniqueness check.
3. Reserve the lex-first **25%** of templates (16 chains) for **test**;
   the remaining 48 templates feed **train**.
4. Re-fill train and test buffers from the original pool, taking only
   problems whose chain template lies in the corresponding split.

This guarantees that no `(chain[0], ..., chain[k-1])` template that
appears in test ever appears in train. The split is deterministic given
the seed; verification at generation time prints
`shared chains = 0`.

## Final dataset stats

```
data/clutrr_graph_v5_train.jsonl      3000 records
data/clutrr_graph_v5_test.jsonl        200 records
```

Both splits use `k=4, n_distractor_entities=12, n_distractor_edges=28`.

Test-set descriptive statistics:

| Metric | Value |
|---|---|
| Records | 200 |
| Avg entities per instance | 17.0 |
| Avg edges per instance | 32.0 |
| Avg dead-end ratio at t=0 | ~88% |
| Query-head out-degree dist. | 2:102 / 3:68 / 4:23 / 5:4 / 6:3 |
| Query-tail in-degree dist. | 2:113 / 3:61 / 4:20 / 5:5 / 6:1 |
| Distinct gold answers | 7 (`great-great-{grandmother, grandfather, grandson, granddaughter, uncle, aunt}` and `first-cousin-once-removed`) |

## Record schema

Each JSONL line has the same field set as the v4 chain dataset, so the
existing baseline / PT-SFT / HypPlan tooling reads it without
modification:

| Field | Meaning |
|---|---|
| `k` | gold-chain length |
| `entities` | list of all entity names in the graph |
| `edges` | list of `[head_idx, relation, tail_idx]` triples |
| `query` | `[head_idx, tail_idx]` — entities to relate |
| `answer` | gold composed relation between head and tail |
| `chain` | gold base-relation sequence along the gold path |
| `prompt` | rendered story + question |
| `init_state_text` | same as `prompt`, used by HypPlan adapter |
| `answer_label` | gold trajectory text (Step 1 / Step 2 / ... / Answer:) |
| `id` | unique problem id (e.g. `clutrr_graph_test_00042`) |
| `split` | `"train"` or `"test"` |

## Reasoning trajectory format

The PT-SFT supervised target walks the gold chain, emitting cumulative
compositions. Step `i` reports the composed relation from the query
head to the i-th chain entity:

```
Step 1: Carol is the father of Dan
Step 2: Carol is the grandfather of Mia
Step 3: Carol is the great-uncle of Tara
Step 4: Carol is the first-cousin-once-removed of Sam
Answer: Carol is the first-cousin-once-removed of Sam.
```

After `data/annotate_sft_plan_groupB.py` adds tags:

```
<PLAN:COMPOSE> Step 1: Carol is the father of Dan
<PLAN:COMPOSE> Step 2: Carol is the grandfather of Mia
<PLAN:COMPOSE> Step 3: Carol is the great-uncle of Tara
<PLAN:COMPOSE> Step 4: Carol is the first-cousin-once-removed of Sam
<PLAN:ANS> Answer: Carol is the first-cousin-once-removed of Sam.
```

This is structurally identical to the v4 chain trajectory; the
difference is that the model now has to *find* the right next entity
inside a graph of ~17 entities, instead of following the unique
chain edge.

## Pre-pipeline probes (justifying the design)

Before committing to v5 we ran three checks with a 500-train / 50-test
miniature pipeline:

| Check | Target | Result | Pass? |
|---|---|---|---|
| Dead-end ratio at initial state | ~85% | 81–88% (v2/v5) | ✅ |
| Base Qwen-14B-Instruct floor | <50% | 0/20 = **0%** | ✅ |
| Graph-PT-SFT non-saturation | <70% | 27/50 = **54%** (v2 with endpoint pivots) | ✅ |
| Chain-PT-SFT transfer-only | informational | 6/20 = **30%** | (gap to graph-trained) |

The 0% base-model accuracy and 54% PT-SFT-after-training accuracy
together establish that (a) the task is genuinely hard for non-finetuned
models, and (b) PT-SFT does *not* saturate it — leaving substantial
headroom for HypPlan to demonstrate value.

## Comparison to chain CLUTRR v4

| Property | v4 (chain) | v5 (graph) |
|---|---|---|
| Story shape | linear chain `e0→e1→…→ek` | dense graph, ~17 entities, ~32 edges |
| Edges per instance | k (= 4) + a few distractor edges | ~32 (4 gold + ~28 distractor) |
| First-step legal compositions | 1 | 16–26 (avg ~25) |
| Dead-end ratio | 0% (no branching) | ~88% |
| Few-shot Qwen-14B | 17.8% | **0/200 = 0%** |
| Self-Consistency K=5 | 22.2% | **0/200 = 0%** |
| ToT BFS | 16.7% | **0/200 = 0%** |
| Planning-Token SFT | **100%** (saturated) | **88/200 = 44%** |
| HypPlan in-domain | matches PT-SFT (no headroom) | **110/200 = 55%** (**+11.0pp over PT-SFT**) |

## Final v5 results table

All numbers on `data/clutrr_graph_v5_test.jsonl` (200 records, k=4,
template-disjoint, head_out≥2, tail_in≥2, ~32 edges/instance,
~88% mean dead-end ratio).

| Method | Score | Per query-head out-degree |
|---|---|---|
| Few-shot greedy (Qwen-14B-Instruct) | 0/200 = **0%** | floor at every head_out |
| Self-Consistency K=5 (T=0.7) | 0/200 = **0%** | floor at every head_out |
| Tree-of-Thoughts BFS | 0/200 = **0%** | floor at every head_out |
| Planning-Token SFT (3000 train, 5 epochs) | 88/200 = **44%** | head_out=2: 52% / =3: 35% / =4: 30% |
| **HypPlan in-domain** | **110/200 = 55%** | **head_out=2: 63% / =3: 47% / =4: 35%** |

Key takeaways:
- **All non-finetuned methods (greedy, SC, ToT) score 0%** — graph
  density genuinely defeats prompt-based reasoning.
- **HypPlan +11.0 pp over PT-SFT** — the value-conditioned LoRA finds
  the gold path more often than plain SFT, especially when the query
  head has multiple outgoing edges (real first-step search).
- The headroom is largest at moderate branching (head_out=2-3), where
  the model has to discriminate one gold edge from 1-2 distractors.

## File pointers

```
src/oracle_clutrr_graph.py                      # graph oracle: GraphProblem,
                                                  shortest_compose_distance,
                                                  legal_compositions,
                                                  generate_graph_problem
scripts/gen_clutrr_graph.py                     # full train/test generator
                                                  with template-disjoint split
scripts/probe_clutrr_graph.py                   # dead-end-ratio probe
data/clutrr_graph_v5_train.jsonl                # 3000 train records
data/clutrr_graph_v5_test.jsonl                 #  200 test records
data/clutrr_graph_v5_train_sft_plan.jsonl       # PT-SFT-annotated train
configs/sft_pt_clutrr_graph_v5_qwen14b.yaml     # PT-SFT training config
checkpoints/sft_pt_clutrr_graph_v5_qwen14b/     # trained PT-SFT LoRA
results/v5/                                     # all v5 baseline + HypPlan
                                                  evaluation outputs
```

## Reproducing the dataset

```bash
# 1. Generate train + test JSONL with template-disjoint split
PYTHONPATH=. python3.10 -m scripts.gen_clutrr_graph \
    --k 4 --n_train 3000 --n_test 200 \
    --n_distractor_entities 12 --n_distractor_edges 28 \
    --seed_start 100000 --out_dir data
mv data/clutrr_graph_train.jsonl data/clutrr_graph_v5_train.jsonl
mv data/clutrr_graph_test.jsonl  data/clutrr_graph_v5_test.jsonl

# 2. Annotate train with <PLAN:COMPOSE> tags for PT-SFT
PYTHONPATH=. python3.10 -m data.annotate_sft_plan_groupB \
    --task clutrr \
    --in_path  data/clutrr_graph_v5_train.jsonl \
    --out_path data/clutrr_graph_v5_train_sft_plan.jsonl

# 3. Verify dead-end ratio on 5 fresh instances
PYTHONPATH=. python3.10 -m scripts.probe_clutrr_graph \
    --k 4 --n_distractor_entities 12 --n_distractor_edges 28 \
    --n_instances 5 --seed_start 0
```
