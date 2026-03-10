# Near-Real EM Trainer (Qwen2.5-7B + LoRA)

This repository implements a near-real version of your EM-style planning-token training draft:

1. **Initialize planning latents** `t_i` from model hidden states.
2. **M-step** jointly optimize planning-latent prediction and reasoning-step likelihood.
3. **E-step** refresh `t_i` using the updated model and planning head.

The implementation uses:
- PyTorch
- Hugging Face Transformers
- PEFT LoRA
- Accelerate

Planning vectors are produced as:

`t_i = Proj(planning_head(h_i))`

where `planning_head(h_i)` has the same dimension as the model hidden size. The current `Proj` is `IdentityProjection`; future Lorentz constraints should be applied after projection.

## Dataset

This project now uses **only** `whynlp/gsm8k-aug` from Hugging Face:
- Dataset page: [whynlp/gsm8k-aug](https://huggingface.co/datasets/whynlp/gsm8k-aug)
- Schema per sample:
  - `question` (string)
  - `steps` (list of strings)
  - `answer` (string)
- Default splits:
  - training: `train`
  - evaluation: `validation` (override to `test` when needed)

No JSONL preprocessing is required for the default setup.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

```bash
PYTHONPATH=. python scripts/train_em.py --config configs/train.yaml
```

### Run with 4 GPUs (proper DDP)

Use Accelerate to launch 4 processes (one per GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --multi_gpu \
  scripts/train_em.py --config configs/train.yaml
```

What to verify in logs:
- `WORLD_SIZE=4` from `scripts/train_em.py`
- `DDP runtime: world_size=4` from trainer startup
- only rank-0 prints checkpoint save messages

### Smoke run (tiny model + small sample count)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --multi_gpu \
  scripts/train_em.py --config configs/smoke.yaml
```

## Baseline vs Trained Evaluation

Train non-planning SFT baseline first:

```bash
PYTHONPATH=. python scripts/train_sft_baseline.py \
  --config configs/train.yaml \
  --output_dir outputs/sft-baseline
```

Evaluate SFT mode (no planning vectors):

```bash
PYTHONPATH=. python scripts/eval_baseline.py \
  --config configs/train.yaml \
  --mode sft \
  --adapter_path outputs/sft-baseline/lora_adapter \
  --output outputs/eval/sft_model.jsonl
```

Evaluate EM mode (dynamic planning vectors at generation time):

```bash
PYTHONPATH=. python scripts/eval_baseline.py \
  --config configs/train.yaml \
  --mode em \
  --adapter_path outputs/em-qwen2.5-7b/em_iter_2_step_0/lora_adapter \
  --output outputs/eval/em_model.jsonl
```

To evaluate on test split:

```bash
PYTHONPATH=. python scripts/eval_baseline.py \
  --config configs/train.yaml \
  --mode sft \
  --split test \
  --adapter_path outputs/sft-baseline/lora_adapter \
  --output outputs/eval/sft_model_test.jsonl
```

The evaluator prints aggregate metrics and writes per-sample predictions:
- `exact_match` (normalized exact string match against reference `answer`)
- `reference_substring_match` (reference answer appears in prediction)
- `last_number_match` (last numeric value matches; useful for math tasks)

## Notes on Losses

- `loss_plan`: MSE proxy over planning latent vectors (`t_i`) as a practical surrogate for the pseudo-code likelihood term.
- `loss_reason`: LM cross-entropy over reasoning step `r_i`, conditioned on `x`, prior steps `r_<i`, and a virtual embedding token directly from projected `t_i` (no extra latent-to-hidden mapper).

## Practical Config Tips

- For real Qwen2.5-7B training, keep LoRA enabled and tune `gradient_accumulation_steps`/`mixed_precision` for your GPU.
- For quick validation, temporarily set `model_name_or_path` to a tiny causal LM (for example, `sshleifer/tiny-gpt2`) before switching back to Qwen2.5-7B.
- To train on only part of the split, set `train_subset_ratio` in config (for example `0.1` for 10%); keep `train_subset_seed` fixed for reproducible sampling.
- `max_train_samples` is still supported; ratio sampling is applied on top of the loaded samples.
- In `--mode em`, evaluator loads `planning_head.pt` with the LoRA adapter and injects `t_0` before first token, then refreshes planning vector after each generated `.`.

## Output Checkpoints

Checkpoints are written under `output_dir`, including:
- LoRA adapter weights
- tokenizer files
- planning head weights
