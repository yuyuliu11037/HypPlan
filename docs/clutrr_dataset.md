# CLUTRR — kinship reasoning task

## What is CLUTRR

CLUTRR (Sinha et al. 2019, "CLUTRR: A Diagnostic Benchmark for Inductive
Reasoning from Text") is a benchmark for **multi-hop kinship reasoning**.

Each problem gives:
- A small story listing family facts (e.g. "Olivia is the father of Sam.")
- A query asking how two named entities are related

The model must compose the chain of facts in the story step by step to
derive the queried kinship relation.

The *difficulty knob* is **k**: the number of edges (facts) that must be
chained to answer the query. Larger k means deeper composition.

In our pipeline we use a custom in-house generator (`data/generate_data_clutrr.py`)
that mirrors CLUTRR semantics but lets us control k per split, add
distractor entities/edges, and pick fresh seeds for held-out evaluation.

## Sample record

A single record from `data/clutrr_train.jsonl`:

```json
{
  "id": "clutrr_train_0",
  "k": 2,
  "entities": ["Olivia", "Sam", "Xavier", "Zach", "Noah"],
  "edges": [
    [3, "daughter", 0],
    [0, "father",   1],
    [4, "father",   0],
    [1, "father",   2]
  ],
  "query": [0, 2],
  "answer": "grandfather",
  "chain": ["father", "father"],
  "prompt": "Zach is the daughter of Olivia.\nOlivia is the father of Sam.\nNoah is the father of Olivia.\nSam is the father of Xavier.\n\nHow is Olivia related to Xavier?",
  "answer_label": "Step 1: Olivia is the father of Sam\nStep 2: Olivia is the grandfather of Xavier\nAnswer: Olivia is the grandfather of Xavier."
}
```

Schema fields:

| Field | Meaning |
|---|---|
| `k` | hop count (composition depth) |
| `entities` | list of entity names; indices 0..n-1 reference these |
| `edges` | facts in the story, each `[src_idx, relation, dst_idx]` |
| `query` | `[head_idx, tail_idx]` — derive how `entities[head_idx]` is related to `entities[tail_idx]` |
| `answer` | gold kinship term (single string, e.g. `"grandfather"`) |
| `chain` | the relation sequence used along the gold composition path |
| `prompt` | natural-language story + question (model input) |
| `answer_label` | gold reasoning trajectory (model target / supervision) |

The problem above asks: how is **Olivia** related to **Xavier**?
Following the chain in the facts:
`Olivia →(father)→ Sam →(father)→ Xavier`
Composing two `father` edges gives `grandfather`. The `answer` field is
the single token `"grandfather"`.

The `prompt` may contain extra distractor edges (in v2/v3) — facts that
don't lie on the gold composition path. The model must select the
relevant edges and ignore the rest.

## Gold reasoning trajectory format

Each step in `answer_label` shows one composition. Intermediate states are
written as natural-language facts about the head entity:

```
Step 1: Olivia is the father of Sam
Step 2: Olivia is the grandfather of Xavier
Answer: Olivia is the grandfather of Xavier.
```

For deeper k=4 problems the trajectory has 4 step lines:

```
Step 1: Ivy is the father of Bob
Step 2: Ivy is the grandfather of Ben
Step 3: Ivy is the great-uncle of Zach
Step 4: Ivy is the first-cousin-once-removed of Xavier
Answer: Ivy is the first-cousin-once-removed of Xavier.
```

The answer line repeats the final relation in a self-contained sentence so
the answer parser can extract it cleanly.

## Versioning history (v1 → v2 → v3)

The dataset has been regenerated three times in this project as we
discovered methodological problems with each prior version. Backups of
older versions are kept under `data/clutrr_*.v1.jsonl` and
`data/clutrr_*.v2.jsonl`.

### v1 — initial generator, no distractors, mixed k

| | train | val | test |
|---|---|---|---|
| n | 2000 | 200 | 200 |
| k | 2,3,4 (≈667 each) | 2,3,4 (≈67 each) | 2,3,4 (≈67 each) |
| distractors | 0 | 0 | 0 |

**Problem:** v1 was 100% memorisable. Train and test drew from the same
k distribution; without distractors every fact in the story is on the
gold chain. The number of distinct (chain-template → answer-relation)
mappings is small enough that a finetuned 14B model can lookup-table
the entire space. PT-SFT and HypPlan both hit 100% on v1, which means
the metric can't distinguish reasoning from memorisation.

### v2 — held-out depth + distractors

| | train | val | test |
|---|---|---|---|
| n | 2000 | 200 | 200 |
| k | **2, 3** | 2, 3 | **4 only** |
| distractors | 2 entities + 2 edges | 2 + 2 | 2 + 2 |

We tightened generalisation by:
1. Holding out depth k=4 entirely from training (train only sees 2-hop and 3-hop chains).
2. Adding 2 distractor entities and 2 distractor edges per story so not
   every edge is on the gold path.

**Problem:** v2 became too hard. Trained on k≤3, the model never saw
4-step compositions; at test it couldn't extend chains into the deeper
kinship vocabulary ("first-cousin-once-removed", "great-great-aunt", …)
that only arises at k=4. HypPlan in-domain dropped to **8% on v2 test** —
not because reasoning failed but because the test distribution was
out-of-domain relative to training.

This was a reasonable held-out-depth probe but not the in-domain
benchmark we needed for the paper. Methodological note: *for an
in-domain test, the test depth must appear in the training mix.*

### v3 — in-domain, fresh-seed test split (current)

| | train | val | test |
|---|---|---|---|
| n | 3000 | 300 | 100 |
| k | **2, 3, 4** (1000 each) | 2, 3, 4 (100 each) | **4 only** |
| distractors | 2 entities + 2 edges | 2 + 2 | 2 + 2 |

Changes from v2:
1. **k=4 returned to training** (1000 of the 3000 train problems are k=4).
2. Test stayed at k=4 only — the hardest depth, where reasoning matters
   most.
3. Distractors kept at 2/2 to prevent trivial lookup memorisation.
4. Test problems use `seed_base + 200_000`; train uses `seed_base`. The
   different seeds produce different specific problems even within the
   same k.

**Verification:** train and test are problem-disjoint — different RNG
seeds produce different `(entities, edges, query)` tuples; we sanity-check
that no `(entities, edges, query)` triple appears in both.

This is the configuration used for all CLUTRR results in the paper.

## How v3 maps to the result tables

In the paper's CLUTRR row we report the methods evaluated on
`data/clutrr_test.jsonl` (100 records, k=4, distractors):

| Method | Score on v3 test |
|---|---|
| Few-shot greedy (Qwen-14B-Instruct) | 1/100 = 1.0% |
| Self-Consistency (K=5 majority) | 0/100 = 0.0% |
| Tree-of-Thoughts (BFS top-1) | 0/100 = 0.0% |
| Planning-Token SFT | (pending — needs v3 retrain) |
| **HypPlan in-domain (1-epoch)** | **99/100 = 99.0%** |

The base 14B model gets the first 3 composition steps mostly right but
arrives at the wrong final relation (e.g. "grandfather" instead of
"first-cousin-once-removed") because it doesn't know the rare deep-chain
kinship vocabulary. HypPlan, trained on the in-domain k∈{2,3,4} chains
including this vocabulary, derives them correctly.

## File map

```
data/generate_data_clutrr.py              # generator (v3 default)
data/clutrr_{train,val,test}.jsonl        # current v3 records
data/clutrr_*.v1.jsonl                    # v1 backup
data/clutrr_*.v2.jsonl                    # v2 backup
data/clutrr_train_sft_plan.jsonl          # PT-SFT-annotated v3 train
data/clutrr_trees_qwen14b/{train,val,test}/   # cached Qwen-14B hidden
                                                states for HypPlan head
data/clutrr_trees_qwen14b.v2/             # v2 cache (kept for reference)
src/oracle_clutrr.py                      # problem semantics + scorer
src/dagger_ood_adapters.py:CLUTRRAdapter  # rollout interface
src/tot_ood.py:CLUTRRAdapter              # ToT propose/value adapter
src/score_ood.py:score_clutrr             # evaluation parser/scorer
```
