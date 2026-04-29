# N-Queens — combinatorial CSP task

## What is N-Queens

The classic N-Queens puzzle: place N queens on an N×N chessboard, one per row,
such that no two queens share the same column or the same diagonal. We use
N-Queens as a constraint-satisfaction-problem benchmark: the model has to do
sequential search with backtracking-style reasoning. Unlike CLUTRR (kinship
composition) or ProofWriter (forward-chaining deduction), N-Queens has a
combinatorial branching factor that grows with N.

The two difficulty knobs we use:

| Knob | Meaning |
|---|---|
| **N** | Board size (5–8 in our setup). More queens = exponentially larger search tree. |
| **k** | Number of queens pre-placed by the prompt. The model places the remaining N−k. |

A k=0 problem is a fresh empty N×N board (hardest — the model picks every queen).
A k=N−1 problem has only the last queen to place (trivial).
Intermediate k values give a mid-difficulty regime.

## Sample record

A single record from `data/nqueens_train.jsonl`:

```json
{
  "id": "nqueens_train_00057",
  "N": 8,
  "k": 3,
  "prefix": [6, 3, 1],
  "gold_extension": [6, 3, 1, 8, 4, 2, 7, 5]
}
```

Schema fields:

| Field | Meaning |
|---|---|
| `id` | unique problem id |
| `N` | board size |
| `k` | number of pre-placed queens |
| `prefix` | length-k 1-indexed column list. `prefix[i]` is the column of the queen pre-placed in row i+1 |
| `gold_extension` | a full lex-min valid placement starting with `prefix`, length N. `gold_extension[i]` is the column of row i+1 |

Notes:
- All indexing is **1-indexed** (rows 1..N, columns 1..N).
- A problem may have many valid completions; `gold_extension` is the lex-smallest one for reproducibility.
- For k=0 records, `prefix` is the empty list and `gold_extension` is the lex-min full solution.
- The pre-placement is guaranteed to admit at least one valid completion (we filter dead-end prefixes during data generation).

## Reasoning trajectory format

The supervised target (Stage-2 DAgger gold and PT-SFT target) is a step-by-step
walk that fills each remaining row in order:

```
Step 1: Place queen in row 4 at column 8.
Step 2: Place queen in row 5 at column 4.
Step 3: Place queen in row 6 at column 2.
Step 4: Place queen in row 7 at column 7.
Step 5: Place queen in row 8 at column 5.
Solution: [6, 3, 1, 8, 4, 2, 7, 5]
```

Conventions:
- "Step N" = the model's **N-th action**, not the row index. For a k=3 prefix problem the first action ("Step 1") places at row 4.
- Each Step states the row and column being filled.
- The final line is `Solution: [c1, ..., cN]`, the full N-tuple of column placements (including the prefix). This is the parsed target for scoring.

The user message in the eval prompt conveys the prefix info ("Already-placed queens: row 1 col 6, …"); the assistant continues with `Step 1: Place queen in row 4…` etc. There is no rendering of prefix queens as Step lines in the assistant turn — the model is trained to start at Step 1 = first new action.

## Train / test split design

The universe of distinct (N, k, prefix) problems at each N is enumerable and
finite (it's the set of all valid prefixes that extend to a full valid placement):

| N | Total distinct (k, prefix) tuples (k∈{0..N−1}) |
|---|---|
| 5 | ~36 |
| 6 | ~21 |
| 7 | ~210 |
| 8 | ~750+ |

We use a **universe-partition split** at the test size N=8 to guarantee
zero train/test problem overlap:

1. Enumerate every valid (k, prefix) tuple for N=8 with k ∈ {0,1,2,3,4}.
2. Reserve a fixed-seed subset for **test**:
   - 12 problems each at k=1, k=2, k=3, k=4 (held out)
   - 1 problem at k=0 (the empty board — only one such problem exists)
   - Total: 45 test records
3. The remaining N=8 problems go to train.
4. For N ∈ {5, 6, 7} (training-only sizes), all valid problems go to train.

This guarantees: every test (N, k, prefix) tuple is **disjoint** from train.
The split is deterministic given the seed.

## Final dataset stats

```
Train: 294 records   {N=5: 34, N=6: 16, N=7: 107, N=8: 137}
Val:   15 records    (small, sampled from train pool, used only for head-trainer logging)
Test:  45 records    (all N=8; k=0:1, k=1:8, k=2:12, k=3:12, k=4:12)
```

The train mix is intentionally biased toward larger N (more search-tree variety).
The test is N=8-only because that's where the search problem is most non-trivial —
small-N problems mostly trivialize once the prefix is given.

## Why these design choices

- **Vary N in train, fix N=8 in test**: lets us train on a curriculum of
  CSP sizes but evaluate the hardest one. Small-N problems contribute
  branching-factor variety to the training distribution.
- **Mixed k in test (k=0…4)**: gives baselines a fair chance — k=4 alone
  would be 4-row search and small enough that a base 14B model could
  occasionally luck into a solution; k=0..1 require almost full search and
  separate the methods.
- **Universe partition (not random shuffle)**: with a fully-enumerable
  problem space, random sampling could trivially leak the same
  (N, k, prefix) tuple into both train and test. Partitioning the universe
  guarantees no leakage.

## Final results

All numbers measured on the same 45-record N=8 test set with answer-accuracy
scoring (parse `Solution: [c1, …, cN]` and validate the placement is conflict-free
and matches the prefix):

| Method | Score | Per-k highlights |
|---|---|---|
| Few-shot greedy (Qwen-14B-Instruct) | 4/45 = 8.9% | k=4: 3/12 |
| Self-Consistency (K=5 majority) | 5/45 = 11.1% | k=4: 4/12 |
| Tree-of-Thoughts (BFS top-1) | 1/45 = 2.2% | k=3: 1/12 |
| Planning-Token SFT | 4/45 = 8.9% | k=4: 3/12, k=0: 1/1 |
| **HypPlan in-domain** | **12/45 = 26.7%** | **k=3: 6/12 = 50%** |

Key takeaways for the paper:
- **HypPlan +15.6 pp** over the best baseline (SC) and **+17.8 pp** over PT-SFT.
- Plain SFT (PT-SFT) doesn't beat the few-shot baseline on this task — the
  combinatorial search isn't well-served by surface-imitation alone.
- HypPlan is strongest at k=3 (50%) where the search tree is mid-sized. At
  k=0 (full empty board) and k=4 (only 4 rows to fill), all methods are
  near floor.
- Tree-of-Thoughts BFS underperforms even greedy on this task — its
  propose-and-value architecture struggles with long combinatorial
  branches that need committing to a partial path.

## File map

```
src/oracle_nqueens.py               # Problem, enumerate_tree, solve_lex_min,
                                      parse_solution, score_solution, etc.
data/generate_data_nqueens_full.py  # universe-partition train/val/test gen
data/nqueens_{train,val,test}.jsonl # current data files (294/15/45)
data/nqueens_train_sft_plan.jsonl   # PT-SFT-annotated train set
data/nqueens_trees_qwen14b/{train,val,test}/   # cached Qwen-14B hidden
                                                 states for HypPlan head
src/dagger_ood_adapters.py:NQueensAdapter      # rollout interface
src/tot_ood.py:NQueensAdapter                  # ToT propose/value adapter
src/score_ood.py:score_nqueens                 # scorer used by baselines
src/eval_stage2_answer.py                      # answer-accuracy eval driver
                                                 (used for HypPlan + uniform
                                                 with baseline scoring)
configs/head_nqueens_qwen14b_rank.yaml         # Stage-1 head config
configs/stage2_dagger_nqueens_qwen14b.yaml     # Stage-2 LoRA config
configs/sft_pt_nqueens_qwen14b.yaml            # PT-SFT config
checkpoints/head_nqueens_qwen14b_rank/         # trained Stage-1 head
checkpoints/dagger_stage2_nqueens_indomain/    # trained Stage-2 LoRA + UpProj
checkpoints/sft_pt_nqueens_qwen14b/            # trained PT-SFT LoRA
```
