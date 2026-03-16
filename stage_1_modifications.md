# Stage 1 Modification: Replace AuxDec with Frozen-LLM Reconstruction

## Summary of Change

Remove the AuxDec module entirely. Replace `L_auxdec` with `L_reconstruct`, which uses the **frozen base LLM itself** to reconstruct each reasoning step conditioned on the planning vector `t_i`. Add a `--reconstruct_mode` flag with two options: `contextual` and `isolated`.

The old `L_lm` (single-pass hidden-state addition) is also removed — `L_reconstruct` subsumes it with a strictly stronger signal because the planning vector is injected at the **embedding level** (so all transformer layers attend to it) rather than the last hidden layer (where only `lm_head` sees it).

---

## 1. What to Remove

### 1.1 Delete file
- `src/model/aux_decoder.py` — the entire AuxDec module.

### 1.2 Remove from `src/model/planning_model.py`
- All imports and references to `AuxDec`.
- Any AuxDec instantiation in `__init__`.
- Any AuxDec forward call in `stage1_forward()`.
- Any AuxDec-related output (e.g., returning `auxdec_loss`).

### 1.3 Remove from `src/train_stage1.py`
- AuxDec parameter group from the optimizer.
- `L_auxdec` from the loss computation.
- Any AuxDec checkpoint saving logic.

### 1.4 Remove CLI args
- `--lambda_aux`
- `--auxdec_layers`
- `--auxdec_heads`

### 1.5 Remove from file tree comment
- Remove `aux_decoder.py` from the directory tree in the spec/comments.

### 1.6 Remove from implementation notes
- Remove note 5 ("AuxDec data feeding") and all other AuxDec references in comments/docstrings.

---

## 2. What to Add

### 2.1 New CLI arg

Add `--reconstruct_mode` with choices `contextual` (default) and `isolated`.

```python
parser.add_argument("--reconstruct_mode", type=str, default="contextual",
                    choices=["contextual", "isolated"],
                    help="How to compute reconstruction loss. "
                         "'contextual': two-pass over full sequence with t_i injected at embedding level. "
                         "'isolated': per-step forward passes with only [PLAN]+t_i as prefix.")
```

### 2.2 New Stage 1 forward pass and loss

Replace the entire Stage 1 forward logic. The new flow has two modes:

#### Mode A: `--reconstruct_mode=contextual`

This is a two-pass approach over the full sequence (same structure as the Stage 2 two-pass, but without LoRA):

```
Pass 1 (compute planning vectors, no gradient):
    with torch.no_grad():
        outputs = frozen_model(input_ids, output_hidden_states=True)
        h_plan_i = outputs.hidden_states[-1] at each [PLAN] position
        t_i = Proj(h_plan_i)   # Proj IS differentiable; detach h_plan_i, not t_i

Pass 2 (reconstruct with plan injection, with gradient):
    Construct modified input embeddings:
        input_embeds = frozen_model.get_input_embeddings()(input_ids)
        For each [PLAN] position k:
            input_embeds[batch, k, :] += t_i   # add planning vector at [PLAN] positions
    outputs = frozen_model(inputs_embeds=input_embeds, output_hidden_states=False)
    logits = outputs.logits

L_reconstruct = CrossEntropyLoss(logits, labels)
    where labels are shifted input_ids (standard causal LM)
    masked to ONLY compute loss on step tokens
    (question tokens, [PLAN] tokens, and padding are masked out with label = -100)
```

**Gradient flow**: `L_reconstruct → logits → frozen model layers (frozen, gradients pass through but don't update) → input_embeds at [PLAN] positions → t_i → Proj`. Only Proj weights are updated.

**Why `torch.no_grad()` on Pass 1**: Pass 1 is only used to compute `h_plan_i`. We detach `h_plan_i` before feeding to Proj (equivalently, we can run Pass 1 under `no_grad` and then call `Proj(h_plan_i.detach())` — but note that `t_i` itself must remain in the computation graph for Pass 2 gradients to reach Proj). Concretely:

```python
# Pass 1
with torch.no_grad():
    outputs_pass1 = model(input_ids=input_ids, output_hidden_states=True)
    h_plan = outputs_pass1.hidden_states[-1][:, plan_positions, :]  # detached by no_grad

# Compute planning vectors (this IS differentiable w.r.t. Proj parameters)
t = proj(h_plan)  # t is in the computation graph

# Pass 2
input_embeds = model.get_input_embeddings()(input_ids)  # frozen embedding, but result needs grad
input_embeds = input_embeds.clone()  # avoid in-place modification
for i, pos in enumerate(plan_positions):
    input_embeds[:, pos, :] = input_embeds[:, pos, :] + t[:, i, :]

outputs_pass2 = model(inputs_embeds=input_embeds)
logits = outputs_pass2.logits
```

**Important**: Even though the base model is frozen, Pass 2 must run **with gradients enabled** (do NOT wrap in `torch.no_grad()`). The frozen model's parameters won't be updated (requires_grad=False), but PyTorch still needs to build the computation graph through the frozen layers so that gradients can flow from the loss back to `t_i` (which is in `input_embeds`) and then to Proj.

#### Mode B: `--reconstruct_mode=isolated`

Per-step forward passes where each step only sees `[PLAN]+t_i` as context — no question, no previous steps:

```
Pass 1 (compute planning vectors, identical to contextual):
    with torch.no_grad():
        outputs = frozen_model(input_ids, output_hidden_states=True)
        h_plan_i = outputs.hidden_states[-1] at each [PLAN] position
        t_i = Proj(h_plan_i)

Pass 2 (per-step reconstruction):
    For each step i in the sequence:
        plan_embed = frozen_model.get_input_embeddings()(PLAN_TOKEN_ID) + t_i
        step_embeds = frozen_model.get_input_embeddings()(step_i_token_ids)
        isolated_input = concat([plan_embed, step_embeds[:-1]], dim=seq)  # teacher forcing
        outputs = frozen_model(inputs_embeds=isolated_input)
        logits_i = outputs.logits
        L_i = CrossEntropyLoss(logits_i, step_i_token_ids)  # predict all tokens of step i

    L_reconstruct = mean(L_0, L_1, ..., L_N)  # average over steps
```

**Important**: The per-step forward passes must also run with gradients enabled (same reasoning as contextual mode).

**Batching strategy for isolated mode**: Each training sequence has a variable number of steps (typically 3–15). The simplest approach:
1. Collect all (t_i, step_i_tokens) pairs across the batch into a flat list.
2. Pad step tokens to the longest step in the batch (or a fixed `max_step_len=256`).
3. Stack into a single batched forward pass through the frozen model: batch dimension = total_steps_in_batch.
4. This is ONE forward pass, not N sequential ones. The "per-step" description above is logical, not literal.

**Memory concern**: If a micro-batch of 2 sequences has 10 steps each, the isolated-mode Pass 2 runs a forward pass with effective batch size 20, each of length ≤256. This is much shorter than the full 2048-token sequences, so memory is comparable. If it overflows, reduce `max_step_len` or process steps in chunks.

### 2.3 Updated Stage 1 loss

```
L_stage1 = L_reconstruct + λ_seg * L_structural_seg + λ_depth * L_structural_depth
```

- `L_reconstruct`: CE loss from the frozen LLM (contextual or isolated mode).
- `L_structural_seg` and `L_structural_depth`: unchanged from current implementation.
- Defaults: `λ_seg = 0.1`, `λ_depth = 0.1`.

Note: there is no `λ_reconstruct` — it is always weight 1.0 (the primary loss).

### 2.4 Updated optimizer param groups

Trainable modules in Stage 1 are now:
- `Proj`
- Structural heads (segment classifier or contrastive head, depth regressor or monotonicity head)

That's it. No AuxDec.

### 2.5 Updated checkpoint saving

Stage 1 saves: `{proj.pt, structural_heads.pt}`. No `aux_heads.pt` or `auxdec.pt`.

---

## 3. Updated CLI

```bash
# Stage 1 with contextual reconstruction (default)
deepspeed --num_gpus=4 src/train_stage1.py \
    --data_path data/prm800k_annotated.jsonl \
    --model_name Qwen/Qwen2.5-7B \
    --proj_type mlp \
    --reconstruct_mode contextual \
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

# Stage 1 with isolated reconstruction (experiment)
deepspeed --num_gpus=4 src/train_stage1.py \
    --data_path data/prm800k_annotated.jsonl \
    --model_name Qwen/Qwen2.5-7B \
    --proj_type mlp \
    --reconstruct_mode isolated \
    --structural_loss simple \
    --lambda_seg 0.1 \
    --lambda_depth 0.1 \
    --max_seq_len 2048 \
    --max_step_len 256 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --lr 2e-4 \
    --output_dir checkpoints/stage1_isolated \
    --deepspeed configs/deepspeed_config.json
```

Note the new `--max_step_len` arg (default 256) is only used in isolated mode for padding/truncating individual steps.

---

## 4. Updated File Tree

```
src/
├── data/
│   ├── dataset.py              # PlanningTokenDataset, collate_fn
│   └── preprocessing.py        # Derive is_boundary, depth; tokenize; build step_ids
├── model/
│   ├── planning_model.py       # PlanningQwen: wraps base model + Proj, custom forward
│   │                           #   - stage1_forward(): two-pass, frozen LLM reconstruction
│   │                           #     supports contextual and isolated modes
│   │                           #   - stage2_forward(): two-pass, embedding-space injection
│   └── proj.py                 # Linear/MLP Proj implementations
├── losses/
│   ├── simple_structural.py    # CE + MSE
│   └── contrastive_structural.py  # InfoNCE + hinge
├── evaluation/
│   ├── baseline_eval.py        # CoT baseline evaluation of vanilla Qwen2.5-7B
│   ├── planning_eval.py        # Autonomous inference with [PLAN] hook
│   └── math_grading.py         # Sympy-based answer matching
├── train_stage1.py             # Stage 1 training loop (frozen base + Proj + structural)
├── train_stage2.py             # Stage 2 training loop (LoRA + Proj, two-pass)
├── configs/
│   ├── deepspeed_config.json
│   └── default_args.yaml
└── scripts/
    ├── run_stage1.sh
    ├── run_stage2.sh
    ├── run_baseline_eval.sh
    └── run_planning_eval.sh
```

---

## 5. Key Implementation Notes

1. **Gradient flow through frozen model**: Both contextual and isolated modes require Pass 2 to run with gradients enabled (`torch.no_grad()` on Pass 1 only). The frozen model's `requires_grad=False` prevents its weights from being updated, but the computation graph must still be built so gradients reach `t_i` → Proj. Verify with a unit test: after `L_reconstruct.backward()`, check that `Proj.weight.grad is not None`.

2. **Memory for contextual mode**: Two full-sequence forward passes through a 7B model. Pass 1 is under `no_grad` (no activation memory). Pass 2 requires activation memory for backprop. With gradient checkpointing on the frozen model + bf16, this should fit. If tight, reduce micro-batch size to 1.

3. **Memory for isolated mode**: Pass 2 batches all steps together (effective batch = total steps across micro-batch). Each step is short (≤256 tokens), so per-sample memory is lower, but the batch count is higher. Monitor and adjust `max_step_len` if needed.

4. **Structural losses are unchanged**: They still operate on `t_i` slices and are computed after Pass 1 (they don't need Pass 2). Their gradients flow through `t_i → Proj` independently of the reconstruction loss.

5. **Stage 2 is completely unchanged by this modification.** Stage 2 never used AuxDec. Its two-pass forward with LoRA remains exactly as specified.

6. **No changes to evaluation scripts.** The evaluation code loads Proj checkpoints; it never referenced AuxDec.