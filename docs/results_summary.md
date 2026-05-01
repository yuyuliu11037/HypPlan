# Cross-task results summary

This is the master tracking table for all 8 canonical tasks evaluated
in the HypPlan paper. **Always uniform within a task** — every method
in a row uses the same test sample (same JSONL, same limit, same IDs).

## Canonical test sets

| # | Task | Test file | Limit / size | Reason this size is canonical |
|---|---|---|---|---|
| 1 | **CLUTRR-Graph v5** | `data/clutrr_graph_v5_test.jsonl` | full = **200** | template-disjoint generation target |
| 2 | **ProofWriter** | `data/proofwriter_test.jsonl` | full = **200** | standard QDep-balanced sample |
| 3 | **N-Queens** | `data/nqueens_test.jsonl` | full = **45** | universe-partition split: *all* valid (k, prefix) tuples at N=8 with k ∈ {0..4} |
| 4 | **ProntoQA (pq)** | `data/prontoqa_test.jsonl` | `--limit 100` | uniform with multimodel sweep (set 2026-05-01) |
| 5 | **Blocksworld (bw)** | `data/blocksworld_test.jsonl` | `--limit 100` | uniform with multimodel sweep (set 2026-05-01) |
| 6 | **Graphcolor (gc)** | `data/graphcolor_test.jsonl` | `--limit 100` | uniform with multimodel sweep (set 2026-05-01) |
| 7 | **G24** | `data/24_test.jsonl` | `--limit 100` (file=956) | standardised since the multimodel sweep — limits compute on a high-variance task |
| 8 | **Rule-chaining (rulechain)** | `data/rulechain_test.jsonl` | `--limit 200` (file=600) | matches the n=200 scale of other tasks |

CLUTRR v4 chain (`data/clutrr_test.jsonl`, 90 records) is **deprecated**
— superseded by CLUTRR-Graph v5.

## Qwen-14B-Instruct results (final-answer accuracy)

Main paper table. All percentages on the canonical test set listed
above with **Qwen-14B-Instruct** as the base model. Empty cells mean
the method has not been evaluated for that task.

| Task | Greedy | SC K=5 | ToT BFS | PT-SFT | OVM | **HypPlan** |
|---|---|---|---|---|---|---|
| CLUTRR-Graph v5 | 0% | 0% | 0% | 44% | 50% | **55%** |
| ProofWriter | 70.5% | 74% | 72% | 49% | 38% | 44.5% |
| N-Queens | 8.9% | 11.1% | 2.2% | 8.9% | 4.4% | **26.7%** |
| ProntoQA (pq) | 74% | 58% | 41% | 49% | — | **75%** |
| Blocksworld | 56% | 66% | 58% | **96%** | 81% | 67% |
| Graphcolor | 63% | 60% | 34% | 64% | 58% | **88%** |
| G24 | 11% | 21% | 1% | 3% | 3% | **57%** |
| Rule-chaining | 60.5% | 77.5% | 22.5% | **88%** | 84% | 80% |

## gpt-oss-20b baselines

Results from `results/multimodel/gptoss20b_*.jsonl`. Same task setup as
the main table, but with **gpt-oss-20b** as the base model instead of
Qwen-14B-Instruct. PT-SFT, OVM, and HypPlan have not been trained /
evaluated for this base model, so those columns are empty.

| Task | Greedy | SC K=5 | ToT BFS | PT-SFT | OVM | HypPlan |
|---|---|---|---|---|---|---|
| CLUTRR-Graph v5 | 0% | 0% | — | — | — | — |
| ProofWriter | 29.5% | 47% | — | — | — | — |
| N-Queens | 0% | 4.4% | — | — | — | — |
| ProntoQA (pq) | 48% | 39% | — | — | — | — |
| Blocksworld | 9% | 2% | — | — | — | — |
| Graphcolor | 51% | 57% | — | — | — | — |
| G24 | 8% | 16% | — | — | — | — |
| Rule-chaining | 15.5% | 52% | — | — | — | — |

## mistral-small-3-24b baselines

Results from `results/multimodel/mistral24b_*.jsonl`. Same task setup as
the main table, but with **mistral-small-3-24b** as the base model.
Coverage is narrower than gpt-oss-20b — only 5 tasks. Same caveat
about empty columns.

| Task | Greedy | SC K=5 | ToT BFS | PT-SFT | OVM | HypPlan |
|---|---|---|---|---|---|---|
| CLUTRR-Graph v5 | 1.5% | 1% | — | — | — | — |
| ProofWriter | 57% | 67.5% | — | — | — | — |
| N-Queens | 11.1% | 11.1% | — | — | — | — |
| ProntoQA (pq) | 81% | 0% (parser bug) | — | — | — | — |
| Blocksworld | 57% | 53% | — | — | — | — |
| Graphcolor | 49% | 49% | — | — | — | — |
| G24 | 8% | 15% | — | — | — | — |
| Rule-chaining | 54% | 74.5% | — | — | — | — |

Sample-size notes:
- Rulechain SC redo at canonical n=200 = 74.5%. Earlier suspect
  (40% on 5 records) was unreliable.
- mistral-24b PQ SC = 0% **persists on redo** (n=100, 0/100). The
  greedy mode gets 81% on the same prompts → confirmed parser bug
  in majority-vote logic for mistral's PQ output format. Not
  re-running again until the parser is fixed.

## Known uniformity gaps (open work)

1. ✅ ~~ProntoQA HypPlan extension to n=200~~ — superseded; canonical PQ
   was lowered to n=100 on 2026-05-01.
2. ✅ ~~Rule-chaining PT-SFT rescore on first n=200~~ — done, 176/200 = 88%.
3. **2026-05-01 canonical change:** BW / PQ / GC canonical lowered
   from n=200 to n=100 to match the multimodel sweep. Qwen entries in
   the main table were re-scored on the first 100 ids of the existing
   200-record JSONL files (no inference re-run). PQ-PT-SFT and PQ-OVM
   doc cells were inconsistent with the actual files: PQ-PT-SFT was
   34% (stale); rescore at n=100 is 49% / at n=200 is 52%. PQ-OVM was
   12% (phantom — no PQ OVM file exists on disk; never trained).
4. **OVM** has not been evaluated on CLUTRR-Graph v5, ProofWriter, PQ,
   or rulechain. (Time-budget-constrained; these are not the prioritised
   tasks for the OVM-vs-HypPlan ablation.)
5. **Few-shot greedy** missing for PQ, BW, GC on Qwen-14B 4-bit. The
   `results/eval_groupB_base/` files only cover proofwriter, clutrr (v4
   deprecated), rulechain (n=30 only), numpath, lineq, synthlogic.
   The `results/multimodel/` greedy files use gpt-oss-20b and
   mistral-24b, not Qwen-14B (different model → not directly
   comparable).

## Procedural rule (for future runs)

Whenever a method-task cell is added to this table, the entry **must**
be on the canonical limit shown in the first table. If it isn't, write
the actual N parenthetically (e.g. "67% (n=100)") and add a row to the
"Known uniformity gaps" section above.

## Where these numbers come from

| Method | Eval script | Score field |
|---|---|---|
| Few-shot greedy | `src/eval_baseline_kpath.py --mode greedy` | `top1_ok` (final-answer parse) |
| Self-Consistency K=5 | `src/eval_baseline_kpath.py --mode sc --K 5 --temperature 0.7` | `majority_ok` |
| ToT BFS | `src/tot_ood.py` | `correct` |
| Planning-Token SFT | `src/eval_pt_ood.py` (re-scored with task scorer) | task-specific |
| OVM | `src/eval_ovm.py --K 4 --beam 3` (re-scored with task scorer) | task-specific |
| HypPlan in-domain | `src/eval_stage2_answer.py` (clutrr/proofwriter/pq/nqueens) or `src/eval_stage2_indomain.py` (bw/gc/rulechain) | `correct` |

For BW and GC the two HypPlan eval scripts give the **same scores**
(verified: strict step-gating == final-answer accuracy on the existing
shards), so numbers are comparable across both.
