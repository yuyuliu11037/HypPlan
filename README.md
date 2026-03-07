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

## Data Input Contract

Current config keeps a placeholder link:

```yaml
dataset_uri: "TODO://path-or-hf-dataset-link"
```

When this placeholder is used and `use_synthetic_if_placeholder_dataset: true`, the code runs a tiny synthetic dataset for smoke testing.

### Required JSONL format (one JSON object per line)

```json
{
  "id": "example-0001",
  "source": "gsm8k",
  "question": "A question string",
  "trajectories": [
    {
      "steps": [
        "Reasoning step 1",
        "Reasoning step 2",
        "Reasoning step 3"
      ]
    },
    {
      "steps": [
        "Alternative trajectory step 1",
        "Alternative trajectory step 2"
      ]
    }
  ]
}
```

`source` must be one of: `gsm8k`, `math`, `aqua`, `svamp`.

### Optional flattened view (for your preprocessing)

You can preprocess into `(question, steps)` trajectories where each trajectory is already split into ordered reasoning steps. The trainer internally flattens record-level trajectories into per-trajectory training paths.

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

### Smoke test without prepared dataset

```bash
PYTHONPATH=. python scripts/train_em.py --config configs/smoke.yaml
```

## Notes on Losses

- `loss_plan`: MSE proxy over planning latent vectors (`t_i`) as a practical surrogate for the pseudo-code likelihood term.
- `loss_reason`: LM cross-entropy over reasoning step `r_i`, conditioned on `x`, prior steps `r_<i`, and a virtual embedding token derived from `t_i`.

## Practical Config Tips

- For real Qwen2.5-7B training, keep LoRA enabled and tune `gradient_accumulation_steps`/`mixed_precision` for your GPU.
- For quick validation, temporarily set `model_name_or_path` to a tiny causal LM (for example, `sshleifer/tiny-gpt2`) before switching back to Qwen2.5-7B.

## Output Checkpoints

Checkpoints are written under `output_dir`, including:
- LoRA adapter weights
- tokenizer files
- planning head weights
- planning-latent-to-hidden projection weights
