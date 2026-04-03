# HypPlan: Hyperbolic Planning Tokens for LLM Reasoning

Train a language model to emit `[PLAN]` tokens at reasoning step boundaries, where each token carries a hyperbolic planning vector that encodes the tree structure of multi-step reasoning.

**Base model**: Qwen/Qwen2.5-7B
**Hardware**: 8x NVIDIA A6000 (48 GB)
**Dataset**: EleutherAI/hendrycks_math (7,500 train / 5,000 test)

## Current Focus: Stage 1 vs Base Model

| Method | Goal | Trainable | Script |
|--------|------|-----------|--------|
| **Stage 1** | Warm up Proj MLP | `Proj` only (LLM frozen) | `scripts/run_stage1.sh` |
| **Base** | Baseline: plain frozen Qwen2.5-7B | None | (no training) |

**Stage 1** teaches the projection MLP to produce planning vectors (same dim as LLM hidden states) that are inserted as virtual tokens before each reasoning step. The LLM is frozen; only ProjMLP is trained.

**Base** is the same frozen Qwen2.5-7B generating without any planning vectors. Used to isolate the effect of Stage 1's planning machinery.

## Project Structure

```
HypPlan/
├── configs/default.yaml        # All hyperparameters
├── src/
│   ├── utils.py                # Step splitting, answer extraction
│   ├── hyperbolic.py           # Lorentz manifold ops
│   ├── projections.py          # ProjMLP (no ProjectBack)
│   ├── model.py                # HypPlanModel wrapper
│   ├── dataset.py              # HF hendrycks_math loader
│   ├── train_stage1.py         # Stage 1 training
│   ├── generate_stage1.py      # Stage 1 inference
│   ├── generate_base.py        # Base model inference (no planning)
│   └── evaluate.py             # Accuracy by level and type
├── scripts/                    # Shell launchers
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train Stage 1
bash scripts/run_stage1.sh

# Evaluate Stage 1
bash scripts/run_eval_stage1.sh

# Evaluate base model (no training needed)
bash scripts/run_eval_base.sh
```

Override defaults with environment variables:

```bash
NUM_GPUS=4 CONFIG=configs/custom.yaml bash scripts/run_stage1.sh
```

## Evaluation

Evaluation extracts the last `\boxed{...}` from each generation and compares to the ground truth. Reports accuracy overall and broken down by MATH level (1-5) and subject type. Uses a 500-example random subset of the test set for fast iteration.
