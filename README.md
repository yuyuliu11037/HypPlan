# HypPlan

Training framework for:

- Two-stage planning-head training on `whyNLP/gsm8k-aug` with `Qwen/Qwen2.5-7B`
- A plain CoT-SFT baseline on the same data

## Features

- Step-aware formatting for `question + reasoning steps + final answer`
- Stage 1 planning training with a frozen backbone and trainable planning MLP
- Stage 2 LoRA finetuning with the planning head still active
- Baseline SFT training without planning vectors
- Checkpoint save/resume, JSONL logging, validation loss, and lightweight generation-based evaluation

## Install

```bash
pip install -r requirements.txt
```

For multi-GPU launch on a 4-GPU node:

```bash
accelerate config default
```

## Training

Planning training:

```bash
python scripts/train_planning.py --config configs/train_planning.yaml
```

Planning training on 4 GPUs:

```bash
accelerate launch --num_processes 4 scripts/train_planning.py --config configs/train_planning.yaml
```

CoT-SFT baseline:

```bash
python scripts/train_sft_baseline.py --config configs/train_sft_baseline.yaml
```

CoT-SFT baseline on 4 GPUs:

```bash
accelerate launch --num_processes 4 scripts/train_sft_baseline.py --config configs/train_sft_baseline.yaml
```

Smoke run:

```bash
python scripts/train_planning.py --config configs/smoke.yaml
```

## Evaluation

```bash
python scripts/eval.py --config configs/train_planning.yaml --checkpoint outputs/planning/latest.pt --mode planning
python scripts/eval.py --config configs/train_sft_baseline.yaml --checkpoint outputs/sft/latest.pt --mode baseline
```

Distributed evaluation on 4 GPUs:

```bash
accelerate launch --num_processes 4 scripts/eval.py --config configs/train_planning.yaml --checkpoint outputs/planning/stage2/latest.pt --mode planning
accelerate launch --num_processes 4 scripts/eval.py --config configs/train_sft_baseline.yaml --checkpoint outputs/sft/baseline/latest.pt --mode baseline
```

## Resume Training

Each stage saves `latest.pt` plus step-numbered checkpoints under `outputs/.../<stage_name>/`.

To resume, set the matching `resume_from` field in the config:

```yaml
training:
  stage1:
    resume_from: outputs/planning/stage1/latest.pt
  stage2:
    resume_from: outputs/planning/stage2/latest.pt
```

For the SFT baseline:

```yaml
training:
  baseline:
    resume_from: outputs/sft/baseline/latest.pt
```

Checkpoint restore includes:

- trainable model parameters
- optimizer state
- scheduler state
- saved step number

## Notes

- The planning path uses an implicit prefix vector rather than a visible planning token.
- Stage 2 trains LoRA adapters plus the planning head.
- Validation generation for planning mode uses step-by-step rollout with the configured decoding limits.
- Multi-GPU training and evaluation are driven by `accelerate launch`.
