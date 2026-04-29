# HypPlan handoff — 2026-04-25 (updated)

Snapshot of the project state for picking up in a fresh conversation.

---

## The hypothesis under test

**"A task-agnostic LoRA, given a meaningful task-specific z signal, will reason better."**

Two pieces:
1. The LoRA is trained only on G24-varied (varied targets within Game-of-24 trees), so it can't memorize the fixed task.
2. The Stage-1 hyperbolic head is trained per-task; at eval time we plug in the matching head and inject z as a virtual token.

If true, we should see `lora + task-z > lora + no-z > base` on OOD tasks.

---

## Final headline results (Qwen-2.5-14B base; 100 records each)

| Mode | G24-100 (in-domain) | ProntoQA | Blocksworld (goal-reaching) | Graph Coloring |
|---|---|---|---|---|
| base (no LoRA, no z) | 11% | 60% | 41% | 61% |
| lora-G24 (no z) | 9% | **63%** | **43%** | **68%** |
| lora-G24 + rand-z | 4% | 43% | 35% | 60% |
| lora-G24 (no z) + CoT prompt | – | 74% | – | – |
| lora-G24 + dense per-step task-z + CoT prompt | – | 67% | – | – |

PT-SFT (separate Qwen + per-task LoRA SFT'd on each task's data, planning-tokens-augmented; not directly comparable):
- G24 **6%** (much worse than base 11%; format-correct planning tokens but arithmetic wrong)
- PQ 52.5% (regresses below base)
- BW 94.5% (memorization of PlanBench gold)
- GC 64% (constraint violations frequent)
- CD 0% strict / 58% lenient (number hallucination)

Tree-of-Thoughts BFS (Qwen-2.5-14B-Instruct, custom step-by-step propose+value, 100 records each):
| Task | Top-1 | Any-of-top-5 | vs base (top1) |
|---|---|---|---|
| PQ | 41% | 44% | base 60% — ToT loses (error compounding on long derivations) |
| BW (goal) | 58% | 83% | base 41% — **+17pp top1, +42pp any-of-K, ToT clearly helps** |
| GC | 34% | 56% | base 61% — value model fails to rank correct colorings |

ToT prompts are custom (not the few-shot prompt from test records), so the comparison has prompt-format confounds. The BW result is robust enough to overcome this; PQ/GC results may partly reflect the prompt change.

HypPlan Stage-1+2 trained IN-DOMAIN per task (DAgger Stage-2 LoRA+UpProjector + per-task Stage-1 head; custom step-by-step prompt; 100 records each):
| Task | Correct | vs base | vs PT-SFT-indom |
|---|---|---|---|
| PQ | **75%** | 60 → +15pp | 52.5 → +22.5pp |
| BW (goal) | **67%** | 41 → +26pp | 94.5 → −27.5pp |
| GC | **88%** | 61 → +27pp | 64 → +24pp |

All three tasks beat the base/LoRA-G24/PT-SFT (excluding BW PT-SFT which is a memorization ceiling). BW jumped 0% → 10% → 67% across v1 → v2 → v3 of the trainer:
- v1 used global-min DDP sync, throwing away ~75% of rollout pairs (only 41 steps/epoch on BW).
- v2 cyclic-pad-to-global-max kept all rollout pairs (266 steps/epoch); BW lifted to 10%.
- v3 added 3 rollouts/problem (~1300 steps/epoch on BW) and per-epoch checkpointing; BW lifted to 67%.

### What the table shows

1. **Dense per-step task-z does NOT help OOD even with a matched CoT prompt.** PQ controlled comparison: `LoRA-G24 + CoT prompt` gives 74% no-z vs 67% with dense per-step task-z (−7pp). The geometric z signal genuinely doesn't transfer to deductive reasoning.
2. **The LoRA itself transfers** to G24-similar tasks even without z. GC (constraint satisfaction with sequential decisions) +7pp over base. PQ +3pp. BW neutral. Most of the in-domain PQ "win" comes from the prompt change (74% no-z, CoT) rather than per-task Stage-2 training.
3. **Random z is universally harmful** — the z channel is "active" in the LoRA but uninformed-z degrades output.

---

## Key checkpoints / artifacts

```
checkpoints/
├── dagger_stage2_24_varied_bal_r4/z_s1234/   # the task-agnostic LoRA
│   ├── lora/
│   ├── up_projector.pt
│   └── config.yaml                            # references head_24_varied_qwen14b_rank
├── head_24_varied_qwen14b_rank/head.pt        # G24 stage-1 head (used by LoRA training)
├── head_pronto_qwen14b_rank/head.pt           # PQ stage-1 head
├── head_blocksworld_qwen14b_rank/head.pt      # BW stage-1 head
├── head_graphcolor_qwen14b_rank/head.pt       # GC stage-1 head
├── sft_pt_cd_qwen14b/lora                     # PT-SFT baselines (per-task SFT)
├── sft_pt_bw_qwen14b/lora
├── sft_pt_pq_qwen14b/lora
├── sft_pt_gc_qwen14b/lora
├── sft_pt_24_qwen14b/lora                     # PT-SFT G24 (added 2026-04-25)
├── dagger_stage2_pq_indomain/                 # Stage-2 in-domain PQ (added 2026-04-26)
├── dagger_stage2_bw_indomain/                 # Stage-2 in-domain BW (added 2026-04-26)
├── dagger_stage2_gc_indomain/                 # Stage-2 in-domain GC (added 2026-04-26)
└── sft_gsm8k_phi15_{baseline,plan}/lora       # Phi-1.5 GSM8K reproduction (verifies PT impl)
```

```
data/
├── 24_varied_{train,val,test}.jsonl           # original 16k varied G24 data
├── 24_varied_bal_{train,val,test}.jsonl       # balanced version (40% target=24) — used for LoRA training
├── prontoqa_test.jsonl                        # 200 records
├── blocksworld_test.jsonl                     # 200 records
├── graphcolor_test.jsonl                      # 200 records (we generated these)
├── pronto_trees_qwen14b/                      # head training caches
├── blocksworld_trees_qwen14b/
├── graphcolor_trees_qwen14b/
└── *_train_sft_plan.jsonl                     # PT-SFT training data (planning-tokens augmented)
```

```
results/
├── eval_g24_indomain/                         # G24-100 4-condition matrix (lora_noz, lora_randz, lora_taskz)
├── eval_pt_ood/                               # PT-SFT eval on PQ/BW/CD/GC
├── eval_pq_dense_z/                           # PQ controlled CoT-prompt no-z vs dense-z (2026-04-26)
├── eval_pt_g24/                               # PT-SFT G24 eval (2026-04-25)
├── tot_ood/{pq,bw,gc}/                        # ToT BFS on the 3 OOD tasks (2026-04-25)
├── eval_stage2_indomain/{pq,bw,gc}/           # HypPlan Stage-1+2 in-domain (2026-04-26)
└── eval_gsm8k_phi15/{baseline,plan}/          # Phi-1.5 GSM8K reproduction
```

---

## Code map (key files added/modified this experiment cycle)

**Oracles + tree-data + state rendering** (per-task):
- [src/oracle_pronto.py](../src/oracle_pronto.py) — forward-chaining proof oracle
- [src/oracle_blocksworld.py](../src/oracle_blocksworld.py) — 4-op blocksworld with state simulation
- [src/oracle_graphcolor.py](../src/oracle_graphcolor.py) — 3-coloring CSP oracle, problem generator
- [data/generate_tree_data_pronto.py](../data/generate_tree_data_pronto.py)
- [data/generate_tree_data_blocksworld.py](../data/generate_tree_data_blocksworld.py)
- [data/generate_tree_data_graphcolor.py](../data/generate_tree_data_graphcolor.py)

**Eval drivers**:
- [src/eval_ood_generic.py](../src/eval_ood_generic.py) — single-shot z (used for v2 results)
- [src/eval_pt_ood.py](../src/eval_pt_ood.py) — PT-SFT inference on OOD tasks
- [src/eval_pt_g24.py](../src/eval_pt_g24.py) — PT-SFT inference on G24 (2026-04-25)
- [src/tot_ood.py](../src/tot_ood.py) — generic ToT BFS for PQ/BW/GC (2026-04-25)
- [src/dagger_ood_adapters.py](../src/dagger_ood_adapters.py) — per-task adapter (2026-04-26)
- [src/dagger_rollout_ood.py](../src/dagger_rollout_ood.py) — generic OOD rollout (2026-04-26)
- [src/train_stage2_dagger_ood.py](../src/train_stage2_dagger_ood.py) — generic OOD Stage-2 trainer (2026-04-26)
- [src/eval_stage2_indomain.py](../src/eval_stage2_indomain.py) — Stage-2 in-domain eval (2026-04-26)
- `configs/stage2_dagger_{pq,bw,gc}_qwen14b.yaml` (2026-04-26)
- [src/eval_gsm8k.py](../src/eval_gsm8k.py) — Phi-1.5 GSM8K eval
- [src/score_ood.py](../src/score_ood.py) — scorers including BW exact-match + goal-reaching simulator, GC validity, PQ letter-match
- [src/generate_24_varied.py](../src/generate_24_varied.py) — G24-100 in-domain eval (uses dense rollout_one)

**Training drivers**:
- [src/train_head.py](../src/train_head.py) — Stage-1 head trainer (DDP gloo)
- [src/train_stage2_dagger_varied.py](../src/train_stage2_dagger_varied.py) — Stage-2 LoRA + UpProjector trainer (gloo, NCCL broken on this host)
- [src/train_sft_24_qwen.py](../src/train_sft_24_qwen.py) — Qwen2.5-14B SFT (gloo DDP)
- [src/train_sft_gsm8k.py](../src/train_sft_gsm8k.py) — Phi-1.5 SFT for the planning-tokens reproduction

**Configs**: `configs/head_*_qwen14b_rank.yaml`, `configs/stage2_dagger_24_varied_balanced.yaml`, `configs/sft_pt_*.yaml` (incl. `sft_pt_24_qwen14b.yaml`), `configs/sft_gsm8k_phi15_*.yaml`

**Docs**: [docs/ood_datasets.md](ood_datasets.md), [README.md](../README.md)

---

## Critical infrastructure notes

1. **NCCL is broken on this host** — always use `backend="gloo"` in `init_process_group`. We have a `HYPPLAN_DIST_BACKEND` env-var convention. See `memory/reference_nccl_topology.md`.
2. **Use `/data/yuyu/.local/bin/torchrun`** (Python 3.10), not the cvlm conda env's python.
3. **Always pre-check `checkpoints/` before training** anything new — see `memory/feedback_precheck_existing_artifacts.md`. (We almost re-trained the CD head before noticing the existing `checkpoints/head_cd_qwen14b_rank/head.pt`.)
4. **Default to minimum-time path** — DDP across multiple GPUs, sharded eval, parallel jobs. See `memory/feedback_minimum_time.md`.
5. **GPU 0 = ruiqi, GPU 5 = jiannan**, generally avoid. GPUs 1-4, 6, 7 are "ours".

---

## Open follow-ups (in suggested priority)

1. **Mixture training**: train a Stage-2 LoRA on a *mixture* of arithmetic + deductive + planning task data, so the LoRA learns to read z across task structures. This is the cleanest test of whether HypPlan's transfer story is achievable at all.
2. **Larger head**: scale Stage-1 head capacity (hidden_dims, hyp_dim) to see if it helps OOD discrimination.
3. **PQ dense-z (CoT prompt) — done**: dense per-step task-z hurts PQ by 7pp under matched-prompt control (74% no-z vs 67% dense-z). Confirms density isn't the bottleneck for OOD task-z; geometric z genuinely doesn't transfer to deductive reasoning.
4. **Cross-validate on more OOD tasks**: arithmetic-similar (Countdown was uninformative due to scoring artifacts; could try arithmetic word problems), or other CSP-style problems (boolean SAT?).

---

## Group B replication — IN PROGRESS (started 2026-04-26)

A second reasoning-family group, mirroring Group A's structure for a publication-strength replication.

| Group A | Group B analog |
|---|---|
| 24_varied_bal (training source) | rulechain (Horn-clause forward chaining, depth {2,3,4}, 16 preds, 18 rules, prefix `p`) |
| ProntoQA | synthlogic (same primitive, depth {5,6,7}, 24 preds, 30 rules, prefix `q` — vocabulary differs from training) |
| Blocksworld | CLUTRR-like (kinship composition, hop counts {2,3,4}, in-house generator) |
| Graph Coloring | mini-Sudoku (4×4 CSP, n_clues {4,5,6}) |

### Code delivered
- Oracles: [src/oracle_rulechain.py](../src/oracle_rulechain.py), [src/oracle_minisudoku.py](../src/oracle_minisudoku.py), [src/oracle_clutrr.py](../src/oracle_clutrr.py). Synthlogic reuses `oracle_rulechain` with `pred_prefix='q'`.
- Adapters added to [src/dagger_ood_adapters.py](../src/dagger_ood_adapters.py): `RuleChainAdapter`, `SynthlogicAdapter`, `CLUTRRAdapter`, `MiniSudokuAdapter`. End-to-end rollout walks 5/5 on each.
- Scorers added to [src/score_ood.py](../src/score_ood.py): `score_rulechain`, `score_clutrr`, `score_minisudoku`. Gold-trajectory round-trip 5/5 on each.
- Eval driver task choices extended in [src/eval_ood_generic.py](../src/eval_ood_generic.py).
- Data generators: [data/generate_data_{rulechain,synthlogic,clutrr,minisudoku}.py](../data/), [data/generate_tree_data_groupB.py](../data/generate_tree_data_groupB.py) (unified, --task dispatch), [data/annotate_sft_plan_groupB.py](../data/annotate_sft_plan_groupB.py) (SFT-PT planning-token annotator).
- Configs: [configs/head_{rulechain,synthlogic,clutrr,minisudoku}_qwen14b_rank.yaml](../configs/), [configs/stage2_dagger_{rulechain_balanced,synthlogic,clutrr,minisudoku}_qwen14b.yaml](../configs/), [configs/sft_pt_{rulechain,synthlogic,clutrr,minisudoku}_qwen14b.yaml](../configs/).
- Vendored CLUTRR source at [external/clutrr/](../external/clutrr/) for reference (not used at runtime; in-house generator avoids matplotlib/sacremoses deps).

### Data delivered
- JSONL: 6000+600+600 rulechain, 1000+200+200 synthlogic, 2000+200+200 clutrr, 2000+200+200 minisudoku.
- SFT-PT JSONL with `<PLAN:APPLY> / <PLAN:COMPOSE> / <PLAN:PLACE> / <PLAN:ANS>` tags.

### Locked-in design (per user feedback 2026-04-26)
- Training depth {2,3,4}, eval depth {5,6,7} (no trivial depth-1)
- Eval predicate prefix `q` (training uses `p`) so model can't generalize via surface vocabulary
- CLUTRR hop counts {2,3,4} for an internal difficulty gradient
- Empirical eval-scale check at depth 5/6/7 with 24 preds / 30 rules: median tree size 23/58/63 nodes, branching 2.0/2.5/2.7, well within enumerable.

### Plan iteration 2 (locked 2026-04-26 PM)

After base 4-bit eval on candidate Group A OOD #1:
- **Linear-Equations: 99% base** — too easy, abandoned (kept code as
  reference under `lineq_*` names).
- **cd_small** (4 numbers, 2-digit target): ~10% base, viable but
  abandoned in favour of numpath.
- **Number-path** (reachability with fixed op set): **34.5% base** —
  selected as Group A OOD #1.
- **mini-Sudoku**: deleted entirely.

Final group composition:
- A (Quantitative / Concrete-Object): 24 Game / Number-path / Blocksworld / Graph Coloring
- B (Logical / Deductive): rule-chaining / ProntoQA / CLUTRR / ProofWriter (CWA d3)

### Three-baseline matrix (locked 2026-04-27 AM)

All 3 × 8 baseline cells run, per-cell results committed under
`results/baselines/{task}_{mode}.jsonl` + `.summary.txt`.

| Task | Base | ToT (top-1) | SC (majority) | PT-SFT |
|---|---|---|---|---|
| 24 Game | 11% | 1% | 21% | 6% |
| Number-path | 34.5% | TBD | 32% | **44.5%** |
| Blocksworld | 41% | 58% | 60% | 94.5% |
| Graph Coloring | 61% | 34% | 66% | 64% |
| rule-chaining | 53% | TBD | 78% | **87.2%** |
| ProntoQA | 60% | 41% | 58% | 52.5% |
| CLUTRR-like | 13% | TBD | 10% | **100%** ⁂ |
| ProofWriter | 70% | TBD | 74% | 49% |

⁂ CLUTRR PT-SFT 100% is a memorisation ceiling — train + test share
the same finite kinship composition table; not a generalisation claim
(parallel to v1 BW PT-SFT 94.5% memorising PlanBench gold).

Caveats:
- Numbers in this session are **4-bit Qwen-14B-Instruct** (memory
  budget); v1 numbers (G24, BW, GC, ProntoQA, the 4 v1 PT-SFT cells)
  are bf16. 4-bit is a lower bound on bf16.
- **ToT cells**: G24 + PQ/BW/GC use paper-faithful BFS
  (`src/tot_baseline.py`, `src/tot_ood.py`). ToT for Number-path /
  rule-chaining / CLUTRR / ProofWriter is **TBD** — earlier numbers in
  these cells were from a non-ToT runner (K-sample structured greedy)
  and were deleted on 2026-04-27. Real ToT adapters for these 4 tasks
  are pending; see "Planned: real ToT on 4 missing datasets" in
  [README.md](../README.md).
- SC reports majority vote (canonical Wang et al. 2023 Self-Consistency).
- rulechain test set is 600 records (vs the 200-record OOD norm)
  because rulechain is the training source and reuses the same test
  generator. SC was capped to 200 via `--limit 200`; PT-SFT used full
  600.

### Pending (HypPlan Stage-1+2 transfer experiment)

- Stage-1 head training for the Group B tasks (rulechain, PQ, CLUTRR,
  ProofWriter; tree-data caches partially built before the plan
  reset).
- Rule-chaining Stage-2 LoRA training (the Group B task-agnostic LoRA).
- Group A and Group B 4-cell OOD eval matrices (base / lora-noz /
  lora-randz / lora-taskz) using the trained LoRAs.

---

## Resume state — 2026-04-28 (end of session)

### What's done since 2026-04-27

- **rulechain in-domain HypPlan**: head + Stage-2 + 6-shard eval done.
  Result: **80% on 200 records** (`results/eval_stage2_indomain/rulechain/`).
- **Phase A1 multimodel sweep** (greedy + SC × 8 tasks × {Qwen-14B,
  GPT-OSS-20B, Mistral-3.2-24B}) — Qwen+GPT-OSS finished; Mistral-24B
  still running on GPU 1 (latest cell: `bw_sc`, ~30 min in at session
  end). Outputs in `results/multimodel/`.
- **Engineering**: efficiency logging (`n_gen_tokens`, `latency_s`)
  baked into [src/eval_baseline_kpath.py](../src/eval_baseline_kpath.py);
  GPT-OSS mxfp4 loader (no BitsAndBytesConfig) added to
  [src/train_sft_pt_qwen.py](../src/train_sft_pt_qwen.py) and
  [src/eval_pt_ood.py](../src/eval_pt_ood.py).
- **GPT-OSS-20B PT-SFT sweep started** on GPU 2 (G24 trained but
  eval skipped — `--task 24` not in eval choices; sweep died there).
  Pending: fix `--choices` in eval driver, restart sweep at G24.
- **CLUTRR v1 → v2 rebuild**: v1 was 100% memorisation. v2 generator
  (`data/generate_data_clutrr.py`) now adds distractor entities/edges
  and supports per-split chain lengths. New default: train k∈{2,3},
  test k=4 (held-out depth) + 2 distractor entities + 2 distractor
  edges. v1 results backed up under `*.v1` filenames; v2 data + tree
  caches regenerated (2400 problems).
- **NCCL recheck**: confirmed still broken post server maintenance.
  6-GPU NCCL run timed out on BROADCAST after 600s with all ranks
  SIGABRT. Falling back to gloo for all DDP. Memory
  `reference_nccl_topology.md` still accurate.
- **CLUTRR v2 head**: 6-GPU gloo, 20 epochs in 7 min,
  `checkpoints/head_clutrr_qwen14b_rank/head.pt` (22 MB).
- **CLUTRR v2 Stage-2 LoRA**: 6-GPU gloo, 2 epochs (~4 h 30 min total),
  `checkpoints/dagger_stage2_clutrr_indomain/lora/` (100 MB) +
  `up_projector.pt` (42 MB). Final avg_loss 0.0040 (epoch 0) → 0.0000
  (epoch 1).

### Currently running at session end

- **GPU 1**: Mistral-3.2-24B Phase A1 SC sweep (cell `bw_sc`,
  PID 3750128). Don't kill — let it continue.
- **GPUs 2–7**: idle (Stage-2 finished).

### Pending in priority order (resume next session)

1. **CLUTRR v2 in-domain HypPlan eval** — 6-shard, 200 test records.
   Command:
   ```bash
   GPUS=(2 3 4 5 6 7)
   for i in 0 1 2 3 4 5; do
     CUDA_VISIBLE_DEVICES=${GPUS[$i]} nohup python3.10 -m src.eval_stage2_indomain \
       --task clutrr \
       --ckpt_dir checkpoints/dagger_stage2_clutrr_indomain \
       --head_path checkpoints/head_clutrr_qwen14b_rank/head.pt \
       --test_data data/clutrr_test.jsonl \
       --output results/eval_stage2_indomain/clutrr/clutrr_shard${i}.jsonl \
       --shard_rank $i --shard_world 6 \
       > logs/clutrr_v2/eval_shard${i}.log 2>&1 &
   done; wait
   cat results/eval_stage2_indomain/clutrr/clutrr_shard*.jsonl \
     > results/eval_stage2_indomain/clutrr/clutrr.jsonl
   ```
   ETA: ~15 min.
2. **CLUTRR v2 PT-SFT (Qwen) retrain + eval** — old v1 LoRA is
   memorisation; need fresh v2 SFT data + retrain. Regenerate
   `data/clutrr_train_sft_plan.jsonl` from v2 train, retrain
   `sft_pt_clutrr_qwen14b/lora`, eval on `data/clutrr_test.jsonl`.
3. **Phase A1 cells 13–14 redo for clutrr v2** + finish proofwriter
   (greedy + SC) on Qwen-14B + GPT-OSS-20B + Mistral-3.2-24B.
4. **Resume indomain chain for numpath + proofwriter**: head +
   Stage-2 + eval each. Use `scripts/run_indomain_chain.sh`. Numpath
   tree-data is cached (2200 problems); proofwriter still needs
   tree-data.
5. **Resume GPT-OSS PT-SFT sweep**: first add `"24"` to eval driver
   `--task` choices (or skip G24 in launcher), then re-launch
   `scripts/run_gptoss_pt_sft_sweep.sh` from G24-eval onward.
6. **Real ToT for 4 missing OOD tasks** (rulechain, clutrr v2,
   numpath, proofwriter): build per-task adapters in
   [src/tot_ood.py](../src/tot_ood.py), then run on Qwen + GPT-OSS +
   Mistral.
7. **Efficiency comparison row** for G24 across all systems.
8. **Final HANDOFF + README** with the full matrix.

### Quick verification commands

```bash
# Confirm CLUTRR v2 artifacts intact
ls -la checkpoints/head_clutrr_qwen14b_rank/head.pt           # 22 MB
ls -la checkpoints/dagger_stage2_clutrr_indomain/lora/        # adapter_model.safetensors 100 MB
ls -la checkpoints/dagger_stage2_clutrr_indomain/up_projector.pt  # 42 MB
wc -l data/clutrr_{train,val,test}.jsonl                      # 2000 / 200 / 200
head -1 data/clutrr_test.jsonl | python3 -c \
  'import json,sys;r=json.loads(sys.stdin.read());print("k=",r["k"])'  # k=4

# Confirm Mistral sweep still alive (or finished)
ps -p 3750128 -o pid,etime,cmd 2>/dev/null
tail -5 logs/multimodel/mistral24b_bw_sc.log 2>/dev/null
```

### Critical context for whoever resumes

- **NCCL is still broken** — every DDP launch must set
  `HYPPLAN_DIST_BACKEND=gloo`. Confirmed 2026-04-28 with a 6-GPU
  test that hung on BROADCAST.
- **GPUs 2–7 are all "ours" right now** — user lifted the
  GPU-restriction, and during this session GPUs 0/5 belonged to
  other users; that may shift. Run `nvidia-smi` first.
- **CLUTRR v1 results were memorisation** (100% on a finite
  composition table). v2 fixes this with held-out depth k=4 +
  distractors. Do NOT cite v1 numbers; they're under `*.v1`.
- **Auto-stop pattern that worked here**: a foreground Bash
  `until grep -q "expected line" log; do sleep 30; done` in
  `run_in_background` mode — fires once when the trigger appears,
  and you get a single notification. Cleaner than a Monitor for
  "wake me at the next save".

---

## Memory (auto-loaded next session)

Already saved in `/data/yuyu/.claude/projects/-data-yuyu-HypPlan/memory/`:
- `project_ood_negative.md` — full results matrix + dense ablation
- `feedback_minimum_time.md` — default to fastest path
- `feedback_precheck_existing_artifacts.md` — grep checkpoints before retraining
- `reference_nccl_topology.md` — NCCL broken; use gloo
- `feedback_plain_language.md` — non-native English speaker; keep it simple
- `feedback_newest_models_only.md` — don't suggest older model fallbacks
- `feedback_ablation_comparisons.md` — prefer both-and not either-or
- Older project context memories (DAgger results, head training history, etc.)

The MEMORY.md index lists them all; they get loaded automatically on session start.

---

## How to resume

1. Read [README.md](../README.md) — the OOD probes section + headline table for results
2. Read this handoff for state
3. If continuing work: the most informative next experiment is **mixture training** (item 1 above)
4. If publishing/writing up: the story is "z signal is in-domain-bound; LoRA's task-skill itself transfers to G24-similar tasks; injection density isn't the bottleneck". Negative for the original transfer hypothesis, partially positive on the LoRA-without-z transfer.

---

## Quick sanity-check commands

```bash
# Verify checkpoints intact
ls checkpoints/dagger_stage2_24_varied_bal_r4/z_s1234/lora
ls checkpoints/head_*_qwen14b_rank/head.pt

# Verify test data intact
wc -l data/{prontoqa,blocksworld,graphcolor}_test.jsonl

# Re-score a result file (uses src/score_ood.py)
python -m src.score_ood --task graphcolor \
  --input results/eval_gc_v1/gc_lora.jsonl

# Re-run a specific eval (sharded across 6 GPUs)
for i in 0 1 2 3 4 5; do
  case $i in 0) GPU=1;; 1) GPU=2;; 2) GPU=3;; 3) GPU=4;; 4) GPU=6;; 5) GPU=7;; esac
  CUDA_VISIBLE_DEVICES=$GPU python -m src.eval_ood_generic \
    --mode lora_taskz \
    --ckpt_dir checkpoints/dagger_stage2_24_varied_bal_r4/z_s1234 \
    --head_path checkpoints/head_graphcolor_qwen14b_rank/head.pt \
    --task graphcolor \
    --test_data data/graphcolor_test.jsonl \
    --output results/eval_gc_v1/gc_lora_taskz_shard${i}.jsonl \
    --shard_rank $i --shard_world 6 &
done; wait
```
