# Stage 1 Training Pipeline: Planning Token with Structural Supervision

## 1. Overview

Train a projection module (Proj) and auxiliary heads on top of a **frozen** Qwen2.5-7B-Base to produce per-step planning tokens that encode both next-step content and hierarchical structure (subgoal segments + within-segment depth). Evaluate against a CoT baseline on MATH test set.

**Hardware**: 4×A100-80G, DeepSpeed ZeRO Stage 2.

---

## 2. Data

### 2.1 Annotated PRM800K

**Path**: `data/prm800k_annotated.jsonl`

Each line is a JSON object representing one full solution trace:

```json
{
  "problem": "What is the greatest common factor of 20! and 200,000?",
  "ground_truth_answer": "40000",
  "steps": [
    {
      "step_index": 0,
      "step_text": "I need to find the GCF of 20! and 200,000.",
      "segment_id": 1,
      "segment_goal": "Set up the problem"
    },
    {
      "step_index": 1,
      "step_text": "Let me start by finding the prime factorization of 200,000.",
      "segment_id": 2,
      "segment_goal": "Factorize 200,000"
    }
  ]
}
```

### 2.2 Derived Labels (computed at data loading time, NOT stored in the file)

From the annotations, derive two labels per step:

- **`is_boundary`**: `True` if this step's `segment_id` differs from the previous step's `segment_id` (or if it's the first step). Boolean.
- **`within_segment_depth`**: integer, 1-indexed position within its segment. E.g., if steps 3,4,5,6 all have `segment_id=3`, their depths are 1,2,3,4.

### 2.3 Data Filtering

- **Discard solutions with 0 steps or only 1 step** (no structure to learn).
- **No correctness filtering for now** (we are ignoring PRM800K correctness labels per earlier discussion; this can be added later as a flag).

### 2.4 Train/Val/Test Split

- Use PRM800K's own split if available, otherwise 90/5/5 random split by problem (not by solution — all solutions for a given problem go into the same split to avoid leakage).
- **Evaluation test set**: the MATH dataset from `qwedsacf/competition_math` on the Hugging Face Hub (same content/schema as the original hendrycks/competition_math). This is separate from PRM800K and used for final answer accuracy only.

---

## 3. Tokenization & Sequence Construction

**Tokenizer**: `Qwen/Qwen2.5-7B` tokenizer (AutoTokenizer).

For each solution trace, construct the training sequence as follows:

```
[Question tokens] [SEP] [Step 0 tokens] [PLAN] [Step 1 tokens] [PLAN] ... [Step N tokens] [EOS]
```

Where:
- `[SEP]` = `\n\n` (literal newline tokens, matching PRM800K step delimiter convention)
- `[PLAN]` = a **reserved special token** added to the tokenizer vocabulary. This is where the planning token embedding `t_i` will be injected. The `[PLAN]` token is placed **between** step `i-1` and step `i`, and its planning vector `t_i` is trained to help predict step `i`.
- The first `[PLAN]` appears between the question and step 0.

### 3.1 Adding the special token

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["[PLAN]"]})
model.resize_token_embeddings(len(tokenizer))
```

### 3.2 Token-level label alignment

For the next-step prediction loss, we need to know which tokens belong to which step. Build a `step_ids` array parallel to the token sequence, where `step_ids[j] = i` means token `j` belongs to step `i`. Tokens in the question get `step_ids[j] = -1`. `[PLAN]` tokens get the step index they are planning FOR (i.e., the next step). This array is used to:
1. Mask the loss to only compute next-token prediction on step tokens (not question tokens).
2. Map each `[PLAN]` position to its structural labels (segment_id, depth).

### 3.3 Max sequence length

Truncate sequences to **2048 tokens**. Discard any solution whose tokenized length exceeds this after truncation would cut mid-step. This keeps things clean.

---

## 4. Model Architecture

### 4.1 Base Model (FROZEN)

`Qwen/Qwen2.5-7B` loaded via `AutoModelForCausalLM`. All parameters frozen throughout Stage 1.

### 4.2 Proj Module (TRAINABLE)

Maps the last hidden state at each `[PLAN]` position to a planning vector `t_i` of the same dimension as the model's hidden size (3584 for Qwen2.5-7B).

**Configurable via `--proj_type` flag:**

- `linear`: single `nn.Linear(hidden_size, hidden_size)`
- `mlp`: `nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))`

The planning vector `t_i` is then **added to** the hidden state at the `[PLAN]` position before the LM head processes subsequent tokens. Concretely:

```
h_plan = base_model.last_hidden_state[plan_position]
t_i = Proj(h_plan)
# Replace the hidden state at the [PLAN] position with h_plan + t_i
# Then continue the forward pass through the LM head
```

Implementation note: this requires a custom forward pass. You cannot just call `model.generate()`. Instead:
1. Run the full model forward to get all hidden states (use `output_hidden_states=True`).
2. Extract hidden states at `[PLAN]` positions.
3. Compute `t_i = Proj(h_plan_i)` for each plan position.
4. Add `t_i` back into the hidden states at plan positions.
5. Pass the modified last hidden layer through the LM head (`model.lm_head`) to get logits.

### 4.3 Auxiliary Heads (TRAINABLE)

#### Simple mode (`--structural_loss=simple`, default):

- **Segment classifier**: `nn.Linear(hidden_size, max_segments)` where `max_segments=16`. Input: `t_i`. Target: `segment_id`. Loss: cross-entropy.
- **Depth regressor**: `nn.Linear(hidden_size, 1)`. Input: `t_i`. Target: `within_segment_depth` (float). Loss: MSE.

#### Contrastive mode (`--structural_loss=contrastive`):

- **Segment contrastive (InfoNCE)**: Split `t_i` into `t_i_seg = t_i[:hidden_size//2]`. For each step `i` in a batch, positive pairs are steps with the same `segment_id` within the same solution, negatives are all other steps. Temperature `τ=0.1`.

```
L_seg = -log( exp(sim(t_i_seg, t_j_seg)/τ) / Σ_k exp(sim(t_i_seg, t_k_seg)/τ) )
```
where `sim` = cosine similarity, `j` shares segment with `i`, sum over all `k` in the batch.

- **Monotonicity (margin-based hinge)**: Split `t_i` into `t_i_depth = t_i[hidden_size//2:]`. For consecutive steps `(i, j)` within the same segment where `depth_i < depth_j`:

```
L_mono = Σ max(0, φ(t_i_depth) - φ(t_j_depth) + margin)
```
where `φ = nn.Linear(hidden_size//2, 1)` (trainable scalar readout), `margin=1.0`.

---

## 5. Loss Function

```
L_total = L_lm + λ_1 * L_structural_1 + λ_2 * L_structural_2
```

- **`L_lm`**: standard causal LM next-token prediction loss, computed ONLY on step tokens (question tokens and [PLAN] tokens are masked out from loss). This is the core loss that teaches the planning token to help predict the next step.
- **`L_structural_1`**: segment loss (CE in simple mode, InfoNCE in contrastive mode).
- **`L_structural_2`**: depth loss (MSE in simple mode, margin hinge in contrastive mode).
- Default: `λ_1 = 0.1`, `λ_2 = 0.1`. Configurable via args.

---

## 6. Training Configuration

```yaml
# DeepSpeed ZeRO Stage 2 config
deepspeed:
  zero_optimization:
    stage: 2
    offload_optimizer:
      device: cpu
  bf16:
    enabled: true
  train_micro_batch_size_per_gpu: 2
  gradient_accumulation_steps: 8
  # Effective batch size: 2 * 8 * 4 GPUs = 64

# Training args
learning_rate: 2e-4
weight_decay: 0.01
warmup_ratio: 0.05
num_train_epochs: 3
max_grad_norm: 1.0
lr_scheduler_type: cosine
save_strategy: epoch
logging_steps: 10
gradient_checkpointing: true  # for the frozen base model, saves memory
seed: 42
```

**Only Proj + auxiliary heads have `requires_grad=True`.** The optimizer only sees these parameters. Use AdamW.

---

## 7. Training Script Structure

```
src/
├── data/
│   ├── dataset.py          # PlanningTokenDataset, collate_fn
│   └── preprocessing.py    # Derive is_boundary, depth; tokenize; build step_ids
├── model/
│   ├── planning_model.py   # Wraps Qwen2.5-7B + Proj + aux heads, custom forward
│   └── proj.py             # Linear/MLP proj implementations
├── losses/
│   ├── simple_structural.py    # CE + MSE
│   └── contrastive_structural.py  # InfoNCE + hinge
├── evaluation/
│   ├── baseline_eval.py    # CoT baseline evaluation of vanilla Qwen2.5-7B
│   ├── planning_eval.py    # Evaluation with external controller injecting planning tokens
│   └── math_grading.py     # Sympy-based answer matching (from PRM800K grading logic)
├── train.py                # Main training loop with DeepSpeed
├── configs/
│   ├── deepspeed_config.json
│   └── default_args.yaml
└── scripts/
    ├── run_train.sh
    ├── run_baseline_eval.sh
    └── run_planning_eval.sh
```

---

## 8. Evaluation

### 8.1 Baseline: Vanilla Qwen2.5-7B CoT

**Goal**: Establish baseline accuracy on MATH test set with standard chain-of-thought prompting.

**Prompt template** (4-shot, following Minerva-style):

```
Solve the following math problem step by step. Put your final answer in \boxed{}.

Problem: {problem}
Solution:
```

**Generation config**:
- `max_new_tokens=1024`
- `temperature=0.0` (greedy)
- `do_sample=False`

**Evaluation**:
- Extract answer from `\boxed{}` in generated text.
- Compare against ground truth using sympy-based grading (port PRM800K's `grader.grade_answer`).
- Report accuracy overall and per difficulty level (Level 1–5) and per subject.

**DDP**: Shard the 500 test problems across 4 GPUs, gather results, compute metrics on rank 0.

### 8.2 Planning Token Model with External Controller

**Goal**: Evaluate the trained Proj model by using an external controller that injects planning tokens at step boundaries during generation.

**External controller inference loop** (on each GPU for its shard of test problems):

```python
for problem in test_shard:
    input_ids = tokenize(prompt + problem)
    all_generated = input_ids

    for step_idx in range(max_steps=20):
        # 1. Insert [PLAN] token
        plan_input = append(all_generated, PLAN_TOKEN_ID)

        # 2. Forward pass through frozen model to get hidden state at [PLAN] position
        with torch.no_grad():
            outputs = model(plan_input, output_hidden_states=True)
            h_plan = outputs.hidden_states[-1][:, -1, :]  # last token = [PLAN]

        # 3. Compute planning vector
        t_i = proj(h_plan)

        # 4. Generate next step: continue generating from the [PLAN] position
        #    with hidden state modified by t_i, until "\n\n" is produced or max tokens
        step_tokens = generate_one_step(
            model, plan_input, t_i,
            stop_string="\n\n",
            max_step_tokens=256
        )

        # 5. Append generated step to context
        all_generated = concat(plan_input, step_tokens)

        # 6. Check if answer is present (\boxed{})
        if contains_boxed(step_tokens):
            break

    # Extract and grade answer
    answer = extract_boxed(all_generated)
    correct = grade_answer(answer, ground_truth)
```

**Critical implementation detail for `generate_one_step`**:
Since we modify hidden states at the `[PLAN]` position, we cannot use `model.generate()` directly. Instead:
1. Run a full forward pass up to and including the `[PLAN]` position.
2. Replace the hidden state at the `[PLAN]` position: `h[-1] = h[-1] + t_i`.
3. Project through `lm_head` to get logits for the next token.
4. Greedily select the next token.
5. Continue standard autoregressive generation (no more modification) until `"\n\n"` or max tokens.

Alternatively, use KV-cache manipulation: after step 2, inject the modified hidden state into the KV cache so that all subsequent tokens attend to the planning-augmented representation. This is more efficient and the **recommended approach**. The implementation should:
- Run the model forward to fill the KV cache up to the `[PLAN]` token.
- Modify the KV cache entries at the `[PLAN]` position to reflect the planning vector.
- Use `model.generate()` with the pre-filled KV cache for the rest of the step.

The exact KV cache modification: for each attention layer `l`, the key and value at the `[PLAN]` position were computed from the original hidden state. We need them recomputed from `h + t_i`. The simplest correct approach: run the forward pass **twice** for the `[PLAN]` token — once to get `h`, compute `t_i = Proj(h)`, then manually set the input embedding at the `[PLAN]` position to `embed([PLAN]) + t_i` and re-run to get the correct KV cache. This is wasteful but correct. A more efficient approach is to only re-run the layers from the point of injection onward, but this requires custom layer-by-layer code.

**Use the same prompt template and generation config as baseline** (minus the [PLAN] injection logic) for fair comparison.

**DDP**: Same sharding as baseline.

### 8.3 Metrics

Report for both baseline and planning model:
- **Overall accuracy** (% of 500 problems correct)
- **Accuracy by difficulty level** (Level 1–5)
- **Accuracy by subject** (7 subjects)
- **Average steps generated** (for planning model)
- **Average segments detected** (for planning model, using the trained segment classifier or contrastive readout)

---

## 9. Key Implementation Notes

1. **Frozen model**: After loading Qwen2.5-7B, call `model.requires_grad_(False)`. Only Proj and aux heads should be in the optimizer's param groups.

2. **Memory**: With ZeRO-2, gradient checkpointing on the frozen base model, bf16, micro-batch=2, the frozen 7B model + KV cache should fit comfortably in 80GB per GPU. The trainable parameters are tiny (<50M).

3. **Custom forward pass**: The core complexity is injecting `t_i` into the hidden states. Implement this as a wrapper class `PlanningQwen` that:
   - Calls the base model with `output_hidden_states=True`
   - Gathers `[PLAN]` positions from `step_ids`
   - Applies Proj to get planning vectors
   - Adds planning vectors to the last hidden layer at those positions
   - Runs `lm_head` on the modified hidden states
   - Returns logits + planning vectors (for auxiliary losses)

4. **Collation**: Sequences in a batch will have different numbers of steps (and thus different numbers of `[PLAN]` tokens). Pad sequences to max length in batch. The `step_ids` array handles masking. The auxiliary losses should gather planning vectors across the batch and compute structure losses on each solution independently (segment/depth labels are per-solution, not cross-solution).

5. **Gradient flow**: `L_lm` gradients flow through `lm_head → modified hidden states → t_i → Proj`. Since the base model is frozen, gradients stop at Proj. The `lm_head` is also frozen; its weights are not updated, but gradients pass through it to reach `t_i`. Verify with a unit test that `Proj.weight.grad` is non-None after `L_lm.backward()`.

6. **Contrastive loss batching**: InfoNCE requires multiple planning vectors. Within a single sequence, there may only be 3–5 planning tokens, giving few contrastive pairs. Gather planning vectors across all sequences in the micro-batch to form a larger pool. Positives = same segment within the same solution. Negatives = everything else. Accumulate across gradient accumulation steps if needed, or just use the micro-batch.

---

## 10. Command-Line Interface

```bash
# Training
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

# Baseline evaluation
torchrun --nproc_per_node=4 src/evaluation/baseline_eval.py \
    --model_name Qwen/Qwen2.5-7B \
    --max_new_tokens 1024 \
    --output_file results/baseline.json

# Planning model evaluation
torchrun --nproc_per_node=4 src/evaluation/planning_eval.py \
    --model_name Qwen/Qwen2.5-7B \
    --proj_checkpoint checkpoints/stage1/proj_best.pt \
    --proj_type mlp \
    --max_steps 20 \
    --max_step_tokens 256 \
    --output_file results/planning.json
```
