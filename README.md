# HypPlan Stage-1 Pipeline

Stage-1 training and evaluation pipeline for planning-token supervision on top of a frozen `Qwen/Qwen2.5-7B`.

## What this repo contains

- Training code for a trainable planning projection module (`Proj`) over a frozen base LM.
- Structural supervision losses:
  - `simple`: segment classification + depth regression
  - `contrastive`: InfoNCE segment loss + monotonic hinge depth loss
- Evaluation scripts for:
  - Vanilla CoT baseline on MATH test
  - Planning-token controller inference on MATH test

## Project layout

- `src/data/`: preprocessing + dataset + collate
- `src/model/`: planning wrapper and projection modules
- `src/losses/`: structural losses
- `src/evaluation/`: baseline/planning evaluation + math grading
- `src/train.py`: main Stage-1 training entrypoint
- `configs/`: DeepSpeed + default args
- `scripts/`: convenience shell scripts

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Core libraries include `torch`, `transformers`, `datasets`, `deepspeed`, and `sympy`.

## Data

Expected training file:

- `data/prm800k_annotated.jsonl`

Each line should contain:

- `problem` (string)
- `ground_truth_answer` (string, optional for training)
- `steps` (list of step objects with `step_text`, `segment_id`, and metadata)

The loader derives:

- `is_boundary`
- `within_segment_depth`

Filtering behavior:

- Drops solutions with 0 or 1 step
- Drops samples whose built sequence would exceed `max_seq_len` (default `2048`)

## Training

Recommended command (4 GPUs):

```bash
deepspeed --num_gpus=4 src/train.py \
  --data_path data/prm800k_annotated.jsonl \
  --model_name Qwen/Qwen2.5-7B \
  --proj_type mlp \
  --structural_loss simple \
  --lambda_seg 0.1 \
  --lambda_depth 0.1 \
  --max_seq_len 2048 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_epochs 3 \
  --lr 2e-4 \
  --output_dir checkpoints/stage1 \
  --deepspeed configs/deepspeed_config.json
```

Or use:

```bash
./scripts/run_train.sh
```

## Baseline evaluation (vanilla CoT)

```bash
torchrun --nproc_per_node=4 src/evaluation/baseline_eval.py \
  --model_name Qwen/Qwen2.5-7B \
  --max_new_tokens 1024 \
  --output_file results/baseline.json
```

Or:

```bash
./scripts/run_baseline_eval.sh
```

## Planning evaluation

```bash
torchrun --nproc_per_node=4 src/evaluation/planning_eval.py \
  --model_name Qwen/Qwen2.5-7B \
  --proj_checkpoint checkpoints/stage1/proj_best.pt \
  --proj_type mlp \
  --structural_loss simple \
  --max_steps 20 \
  --max_step_tokens 256 \
  --output_file results/planning.json
```

Or:

```bash
./scripts/run_planning_eval.sh
```

## Outputs

- Checkpoints:
  - `checkpoints/stage1/epoch_*.pt`
  - `checkpoints/stage1/proj_best.pt`
- Metrics/results:
  - `results/baseline.json`
  - `results/planning.json`

## Notes

- Base LM parameters are frozen; only `Proj` and structural heads are trainable.
- The tokenizer is extended with `[PLAN]` at runtime.
- For reproducibility, set `--seed` (default `42`).
