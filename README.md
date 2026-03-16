# HypPlan

Two-stage planning-token training pipeline on top of `Qwen/Qwen2.5-7B`.

This repo implements:

- **Stage 1 (warm-up)**: frozen base model + trainable projection (`Proj`) and structural losses, with reconstruction via the frozen LLM itself.
- **Stage 2 (joint tuning)**: LoRA tuning + `Proj` with two-pass plan injection.
- **Evaluation**:
  - LoRA-finetuned CoT baseline (same LoRA config as stage 2, trained on first 8000 samples of `prm800k_raw.jsonl`)
  - planning model inference (`autonomous` or `external_controller`)

The full design spec is in `pipeline_spec_v2.md`.

## Repository Layout

- `src/train_stage1.py`: Stage 1 training entrypoint
- `src/train_stage2.py`: Stage 2 training entrypoint
- `src/train_baseline.py`: Baseline LoRA SFT entrypoint (no planning tokens)
- `src/data/`: preprocessing and dataset
- `src/model/`: projection module and planning wrapper
- `src/losses/`: structural loss implementations
- `src/evaluation/`: baseline/planning evaluation and answer grading
- `configs/`: DeepSpeed and default argument configs
- `scripts/`: ready-to-run launch scripts

## Requirements

- Python 3.10+
- CUDA GPUs (scripts are configured for 4 GPUs)
- Install deps:

```bash
pip install -r requirements.txt
```

## Data Format

Stage 1/2 training expects a JSONL file where each line has:

- `problem` (string)
- `ground_truth_answer` (string, optional for training)
- `steps` (list of step objects with `step_index`, `step_text`)
- `annotations` (recommended), where each item includes:
  - `step_index`
  - `step_text`
  - `segment_id`
  - `segment_goal`

Important:

- If `annotations` is missing, the loader falls back to `steps` for structure labels.
- In that fallback case, each `steps` item must already include `segment_id`, otherwise structural label derivation fails.
- Samples with `<=1` step are dropped.

Default training path is `data/prm800k_annotated.jsonl`.

## Create Train/Eval/Test Splits

Generate split files from `data/prm800k_annotated.jsonl`:

```bash
python -m src.data.split_dataset \
  --input_path data/prm800k_annotated.jsonl \
  --output_dir data/prm800k_splits \
  --seed 42 \
  --eval_name eval
```

This creates:

- `data/prm800k_splits/train.jsonl`
- `data/prm800k_splits/eval.jsonl`
- `data/prm800k_splits/test.jsonl`

## Quick Start

Run from repo root.

### Stage 1

```bash
bash scripts/run_stage1.sh
```

### Stage 2

```bash
bash scripts/run_stage2.sh
```

### Baseline Evaluation

Fine-tunes the base model with plain LoRA SFT (first 8000 samples of `data/prm800k_raw.jsonl`, same LoRA config as stage 2), then evaluates. This ensures a fair comparison against the planning model.

```bash
bash scripts/run_baseline_eval.sh
```

### Planning Evaluation (autonomous mode in script)

```bash
bash scripts/run_planning_eval.sh
```

## Direct CLI Examples

The scripts above wrap these commands:

- Stage 1: `deepspeed --num_gpus=4 --module src.train_stage1 ...`
- Stage 2: `deepspeed --num_gpus=4 --module src.train_stage2 ...`
- Baseline SFT: `deepspeed --num_gpus=4 --module src.train_baseline ...`
- Baseline eval: `torchrun --nproc_per_node=4 -m src.evaluation.baseline_eval --lora_adapter_path checkpoints/baseline/lora_adapters ...`
- Planning eval: `torchrun --nproc_per_node=4 -m src.evaluation.planning_eval ...`

For smoke tests on smaller files, override `--data_path` (for example `data/prm800k_annotated_10.jsonl`) and optionally `--limit`.

## Outputs

- Stage 1 (`--output_dir`, default `checkpoints/stage1`):
  - `proj.pt`
  - `structural_heads.pt`
  - `train_args.json`
- Stage 2 (`--output_dir`, default `checkpoints/stage2`):
  - `lora_adapters/`
  - `proj.pt`
  - `plan_token_delta.pt`
  - `train_args.json`
- Baseline SFT (`--output_dir`, default `checkpoints/baseline`):
  - `lora_adapters/`
  - `train_args.json`
- Evaluation:
  - `results/baseline.json`
  - `results/stage2_autonomous.json` (or your selected output path)

## Notes

- The run scripts assume multi-GPU (4 processes). Adjust `--num_gpus` / `--nproc_per_node` for your machine.
- `configs/deepspeed_config.json` sets ZeRO-2 + bf16 defaults.
- The tokenizer is extended with a special token: `[PLAN]`.
- Stage 1/2 scripts use `data/prm800k_splits/train.jsonl` with `--presplit_data`.
- Eval scripts use `data/prm800k_splits/eval.jsonl` by default (`--local_eval_path`).

