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
| lora (no z) | 9% | **63%** | **43%** | **68%** |
| lora + rand-z | 4% | 43% | 35% | 60% |
| **lora + task-z** | **12%** ✅ | 62% | 33% | 67% |
| **lora + task-z (dense)** | (= 12%, same as above) | n/a (no boundaries to inject at) | 36% | 64% |

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
| BW (goal) | **10%** | 41 → −31pp | 94.5 → −84.5pp |
| GC | **88%** | 61 → +27pp | 64 → +24pp |

PQ + GC: HypPlan Stage-1+2 in-domain clearly beats every prior arm. BW: model still mostly cycles (72/100 hit depth budget), but the cyclic-pad fix raised it from 0% → 10%. Initial sync used global-min, throwing away ~75% of the rollout pairs; replacing it with cyclic-pad-to-global-max (each rank repeats its own pairs to match the busiest rank) gives 266 train steps/epoch vs 82 before — 3.2× more supervision and the first non-zero BW solve rate. Same prompt for train + eval; for BW the gap reflects task complexity (multi-block 3D state) on a base model not pretrained for planning.

### What the table shows

1. **Z works in-domain, doesn't transfer OOD.** Only G24-100 has `lora_taskz > lora_noz` (+3pp). On all 3 OOD probes, `lora_taskz ≤ lora_noz`.
2. **Density isn't the OOD bottleneck** — we ran dense per-step-boundary injection on BW (33→36) and GC (67→64). Still fails to beat no-z. So the OOD failure is task-structure mismatch, not injection sparsity.
3. **The LoRA itself transfers** to G24-similar tasks even without z. GC (constraint satisfaction with sequential decisions) +7pp over base. PQ +3pp. BW neutral.
4. **Random z is universally harmful** — the z channel is "active" in the LoRA but uninformed-z degrades output.

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
├── eval_ood_v2/                               # OOD single-shot 4-condition: PQ, BW, GC, CD
├── eval_gc_v1/                                # graph coloring 4-condition + PT-SFT
├── eval_pt_ood/                               # PT-SFT eval on PQ/BW/CD/GC
├── eval_dense_z/                              # dense per-step injection ablation (BW, GC)
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
- [src/eval_dense_z.py](../src/eval_dense_z.py) — **dense per-step z** ablation (new); supports tasks g24/gc/bw, modes single/dense
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
3. **PQ dense z**: PQ outputs are 1-2 tokens with our current prompt, so dense-injection has nowhere to fire. Could change prompt to encourage step-by-step (CoT) and re-run with dense.
4. **Cross-validate on more OOD tasks**: arithmetic-similar (Countdown was uninformative due to scoring artifacts; could try arithmetic word problems), or other CSP-style problems (boolean SAT?).

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
