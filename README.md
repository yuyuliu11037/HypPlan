# HypPlan: Hyperbolic Planning Vectors for LLM Reasoning

Inject a learned planning vector at every reasoning step boundary. At each boundary, the LLM's hidden state is passed through a projection MLP to produce a tangent vector on the Lorentz hyperboloid; this vector is then injected into the embedding sequence as a virtual token to guide the generation of the next reasoning step.

## Tasks

### MATH (hendrycks_math)

**Base model**: Qwen/Qwen2.5-7B
**Dataset**: EleutherAI/hendrycks_math (7,500 train / 5,000 test)

| Method | Goal | Trainable | Script |
|--------|------|-----------|--------|
| **Stage 1** | Warm up Proj MLP | `Proj` only (LLM frozen) | `scripts/run_stage1.sh` |
| **Base** | Baseline: plain frozen Qwen2.5-7B | None | (no training) |

**Stage 1** teaches the projection MLP to produce planning vectors (same dim as LLM hidden states) that are injected into the embedding sequence (via `inputs_embeds`, no vocabulary expansion) at each reasoning-step boundary. The LLM is frozen; only ProjMLP is trained.

**Base** is the same frozen Qwen2.5-7B generating without any planning vectors. Used to isolate the effect of Stage 1's planning machinery.

### Game of 24

**Base model**: meta-llama/Llama-3.1-8B-Instruct
**Dataset**: Custom Game of 24 data (`data/24_*.jsonl`)

| Method | Goal | Trainable | Script |
|--------|------|-----------|--------|
| **Zero-shot / Few-shot** | Baseline: raw LLM prompting | None | `scripts/run_24_zeroshot.sh` |
| **CoT-SFT** | Fine-tune on chain-of-thought trajectories | LoRA adapters | `scripts/run_sft_24.sh` |
| **Planning Vectors** | Inject planning vectors on top of SFT model | `Proj` only (SFT model frozen) | `scripts/run_plan_24.sh` |

**Zero-shot / Few-shot** prompts the base LLM directly without any training.

**CoT-SFT** fine-tunes the base model with LoRA on verified chain-of-thought trajectories for the Game of 24 task.

**Planning Vectors** freezes the SFT-trained model and trains a ProjMLP to insert planning vectors at step boundaries, the same approach as the MATH task.

## Project Structure

```
HypPlan/
├── configs/
│   ├── default.yaml            # MATH hyperparameters
│   ├── sft_24.yaml             # Game of 24 SFT config
│   └── plan_24.yaml            # Game of 24 planning vector config
├── src/
│   ├── utils.py                # Step splitting, answer extraction
│   ├── hyperbolic.py           # Lorentz manifold ops
│   ├── projections.py          # ProjMLP
│   ├── model.py                # HypPlanModel wrapper
│   ├── dataset.py              # HF hendrycks_math loader
│   ├── train_stage1.py         # MATH Stage 1 training
│   ├── generate_stage1.py      # MATH Stage 1 inference
│   ├── generate_base.py        # MATH base model inference
│   ├── evaluate.py             # MATH accuracy evaluation
│   ├── dataset_24.py           # Game of 24 SFT dataset
│   ├── dataset_24_plan.py      # Game of 24 planning dataset
│   ├── train_sft_24.py         # Game of 24 CoT-SFT training
│   ├── train_plan_24.py        # Game of 24 planning vector training
│   ├── generate_24_zeroshot.py # Game of 24 zero/few-shot inference
│   ├── generate_24_sft.py      # Game of 24 SFT inference
│   ├── generate_24_plan.py     # Game of 24 planning vector inference
│   └── evaluate_24.py          # Game of 24 solution validation
├── scripts/                    # Shell launchers
├── data/                       # Game of 24 data files
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

**Hardware**: 8x NVIDIA A6000 (48 GB)

## Usage

### MATH

```bash
# Train Stage 1
bash scripts/run_stage1.sh

# Evaluate Stage 1
bash scripts/run_eval_stage1.sh

# Evaluate base model (no training needed)
bash scripts/run_eval_base.sh
```

### Game of 24

```bash
# Zero-shot / few-shot baseline
bash scripts/run_24_zeroshot.sh [MODEL] [NUM_SHOTS]

# Train CoT-SFT
bash scripts/run_sft_24.sh

# Train planning vectors (requires SFT checkpoint)
bash scripts/run_plan_24.sh
```

Override defaults with environment variables or config path arguments:

```bash
NUM_GPUS=4 CONFIG=configs/custom.yaml bash scripts/run_stage1.sh
bash scripts/run_sft_24.sh configs/sft_24.yaml
```

## Evaluation

**MATH**: Extracts the last `\boxed{...}` from each generation and compares to the ground truth. Reports accuracy overall and broken down by MATH level (1–5) and subject type. Uses a 500-example random subset of the test set for fast iteration.

**Game of 24**: Validates each generated 3-step arithmetic solution by checking that all four input numbers are used exactly once and the final result equals 24.
