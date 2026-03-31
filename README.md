# HypPlan: Hyperbolic Planning Tokens for LLM Reasoning

Train a language model to emit `[PLAN]` tokens at reasoning step boundaries, where each token carries a hyperbolic planning vector that encodes the tree structure of multi-step reasoning.

**Base model**: Qwen/Qwen2.5-7B
**Hardware**: 8x NVIDIA A6000 (48 GB)

## Training Pipeline

| Stage | Goal | Trainable | Script |
|-------|------|-----------|--------|
| **1** | Warm up Proj MLP | `Proj`, `ProjectBack` (LLM frozen) | `scripts/run_stage1.sh` |
| **2** | Align hyperbolic distances to reasoning tree | `Proj`, `ProjectBack`, scalar `c` (LLM frozen) | `scripts/run_stage2.sh` |
| **3** | Joint fine-tuning with LoRA | LoRA adapters + `Proj` + `ProjectBack` | `scripts/run_stage3.sh` |

**Stage 1** teaches the projection MLP to produce planning vectors that help predict the next reasoning step. **Stage 2** adds a tree loss (`L_tree`) so hyperbolic distances between planning vectors match tree distances between reasoning nodes. **Stage 3** applies LoRA to the LLM so it learns to emit `[PLAN]` tokens and use the injected planning vectors.

## Project Structure

```
HypPlan/
├── configs/default.yaml           # All hyperparameters
├── src/
│   ├── data/
│   │   ├── utils.py               # Step splitting, \boxed{} extraction
│   │   ├── dataset_stage1.py      # Correct generations (Stage 1 & 3)
│   │   └── dataset_stage2.py      # All generations + tree metadata (Stage 2)
│   ├── model/
│   │   ├── hyperbolic.py          # Lorentz manifold ops
│   │   ├── proj.py                # ProjMLP + ProjectBack
│   │   ├── plan_model.py          # HypPlanModel wrapper
│   │   └── lora_utils.py          # LoRA setup via peft
│   ├── training/
│   │   ├── stage1.py              # Warm up Proj
│   │   ├── stage2.py              # + tree loss
│   │   ├── stage3.py              # LoRA + two-pass
│   │   └── cot_sft.py             # CoT-SFT baseline (standard LoRA)
│   ├── inference/
│   │   ├── generate.py            # HypPlan generation with [PLAN] hook
│   │   └── generate_cot_sft.py    # Baseline generation (no planning)
│   └── eval/evaluate.py           # Accuracy by level and type
├── scripts/                       # Shell launchers for each stage
└── results/                       # Data files (math_filtered.jsonl, reasoning_trees.jsonl)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Data preparation scripts live in `src/` (run once, outputs already in `results/`):

1. **`sample_math.py`** — Sample 16 generations per MATH problem using Qwen2.5-7B.
2. **`grade_and_filter.py`** — Grade generations and filter to problems with pass rate in [0.1, 0.9].
3. **`build_reasoning_tree.py`** — Build reasoning trees by clustering semantically similar steps.

Outputs:
- `results/math_filtered.jsonl` — 1,493 problems with 16 generations each, graded.
- `results/reasoning_trees.jsonl` — Reasoning trees with pairwise node distances.

## Usage

Run the three training stages in order:

```bash
# Stage 1: Warm up Proj (frozen LLM)
bash scripts/run_stage1.sh

# Stage 2: Add tree structure loss
bash scripts/run_stage2.sh

# Stage 3: LoRA fine-tuning with [PLAN] tokens
bash scripts/run_stage3.sh

# Inference + evaluation
bash scripts/run_eval.sh
```

Override defaults with environment variables:

```bash
NUM_GPUS=4 CONFIG=configs/custom.yaml bash scripts/run_stage1.sh
```

## CoT-SFT Baseline

A standard LoRA fine-tuning baseline on the same correct generations, same compute budget, no planning tokens. Used to isolate the effect of HypPlan's hyperbolic planning machinery.

```bash
# Train
bash scripts/run_cot_sft.sh

# Inference (multi-GPU, respects CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -m src.inference.generate_cot_sft \
    --checkpoint_dir checkpoints/cot_sft \
    --input results/math_filtered.jsonl \
    --output results/eval/cot_sft_generations.jsonl \
    --num_gpus 5

# Evaluate
python -m src.eval.evaluate \
    --input results/eval/cot_sft_generations.jsonl \
    --output results/eval/cot_sft_metrics.json
```

The baseline uses identical LoRA config (r=16, alpha=32, same target modules, lr, epochs) so the only difference is the absence of `[PLAN]` tokens and hyperbolic planning vectors.

## Evaluation

Evaluation extracts the last `\boxed{...}` from each generation and compares to the ground truth. Reports accuracy overall and broken down by MATH level (1-5) and subject type.
