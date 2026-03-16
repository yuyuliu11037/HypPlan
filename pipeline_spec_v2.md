# Two-Stage Training Pipeline: Planning Token with Autonomous Inference

## 1. Overview

Train a planning system in two stages on top of Qwen2.5-7B-Base:

- **Stage 1 (Warm-up)**: Freeze the base model. Train a projection module (Proj), an auxiliary decoder (AuxDec), and structural heads to produce per-step planning tokens that encode next-step content and hierarchical structure (subgoal segments + within-segment depth).
- **Stage 2 (Joint fine-tuning)**: LoRA-tune the base model jointly with Proj so that (a) the model learns to emit `[PLAN]` tokens autonomously and (b) the model learns to consume the injected planning vectors.

At inference, the model autonomously generates `[PLAN]` tokens. A lightweight hook in the generation loop detects `[PLAN]`, runs Proj on the current hidden state, and injects the resulting vector back into the next position's input embedding. No external controller decides when or what to plan.

Evaluate against a CoT baseline on the MATH test set.

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
- **Evaluation test set**: the 500-problem MATH test split (from `hendrycks/competition_math` HuggingFace dataset). This is separate from PRM800K and used for final answer accuracy only.

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

### 4.1 Base Model

`Qwen/Qwen2.5-7B` loaded via `AutoModelForCausalLM`.

- **Stage 1**: All base model parameters **frozen**.
- **Stage 2**: Base model adapted via **LoRA** (trainable), all other base parameters remain frozen.

### 4.2 Proj Module (TRAINABLE in both stages)

Maps the last hidden state at each `[PLAN]` position to a planning vector `t_i` of the same dimension as the model's hidden size (3584 for Qwen2.5-7B).

**Configurable via `--proj_type` flag:**

- `linear`: single `nn.Linear(hidden_size, hidden_size)`
- `mlp`: `nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))`

**Dimensional decomposition of `t_i`**: The planning vector is conceptually split into two halves:
- `t_i_seg = t_i[:hidden_size // 2]` — encodes subgoal identity (which segment this step belongs to).
- `t_i_depth = t_i[hidden_size // 2:]` — encodes refinement level (position within the segment).

This is a logical partitioning only. Proj outputs the full vector; slicing happens at loss computation time.

### 4.3 Injection Mechanism

The injection point differs between the two stages:

**Stage 1 (hidden-state addition, same as original spec)**:
```
h_plan = base_model.last_hidden_state[plan_position]
t_i = Proj(h_plan)
modified_hidden = h_plan + t_i
logits = lm_head(modified_hidden)  # for LM loss gradient to flow back to Proj
```

**Stage 2 (embedding-space addition)**:
```
h_plan = base_model.last_hidden_state[plan_position]
t_i = Proj(stop_gradient(h_plan))
# At the position immediately AFTER [PLAN]:
input_{k+1} = Embed(r_{k+1}) + t_i
```

The embedding-space injection in Stage 2 is necessary because LoRA is adapting the base model: the plan must enter at the input level so all transformer layers (including LoRA-adapted layers) can attend to it. See Section 5.2 for the two-pass training procedure that handles the circular dependency.

### 4.4 AuxDec — Auxiliary Decoder (Stage 1 only, TRAINABLE)

A small transformer decoder that takes `t_i` as a prefix and autoregressively generates the tokens of step `i`. Its purpose is to provide a content-level training signal for Proj during Stage 1, when the base model is frozen and cannot learn to consume planning vectors.

```python
class AuxDec(nn.Module):
    def __init__(self, hidden_size, vocab_size, n_layers=2, n_heads=8):
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)  # project t_i to prefix
        self.layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, t_i, step_token_embeddings, step_token_ids):
        # t_i: (batch, hidden_size) — one planning vector per step
        # step_token_embeddings: (batch, seq_len, hidden_size) — embeddings of step_i tokens
        # step_token_ids: (batch, seq_len) — target token ids for CE loss

        prefix = self.prefix_proj(t_i).unsqueeze(1)  # (batch, 1, hidden_size)
        decoder_input = torch.cat([prefix, step_token_embeddings[:, :-1, :]], dim=1)
        # Apply causal mask
        output = self.layers(decoder_input, memory=None)  # self-attention only
        logits = self.lm_head(output)
        loss = F.cross_entropy(logits.view(-1, vocab_size), step_token_ids.view(-1))
        return loss
```

**Parameter budget**: ~50M parameters (2-layer transformer decoder with hidden_size=3584, 8 heads). This is small relative to the 7B base model.

**AuxDec is discarded after Stage 1.** It never appears in Stage 2 or at inference.

### 4.5 Structural Heads (Stage 1 only, TRAINABLE)

These operate on slices of `t_i` to enforce hierarchical organization.

#### Simple mode (`--structural_loss=simple`, default):

- **Segment classifier**: `nn.Linear(hidden_size // 2, max_segments)` where `max_segments=16`. Input: `t_i_seg = t_i[:hidden_size // 2]`. Target: `segment_id`. Loss: cross-entropy.
- **Depth regressor**: `nn.Linear(hidden_size // 2, 1)`. Input: `t_i_depth = t_i[hidden_size // 2:]`. Target: `within_segment_depth` (float). Loss: MSE.

#### Contrastive mode (`--structural_loss=contrastive`):

- **Segment contrastive (InfoNCE)**: For each step `i` in a batch, positive pairs are steps with the same `segment_id` within the same solution, negatives are all other steps. Temperature `τ=0.1`.

```
L_seg = -log( exp(sim(t_i_seg, t_j_seg)/τ) / Σ_k exp(sim(t_i_seg, t_k_seg)/τ) )
```
where `sim` = cosine similarity, `j` shares segment with `i`, sum over all `k` in the batch.

- **Monotonicity (margin-based hinge)**: For consecutive steps `(i, j)` within the same segment where `depth_i < depth_j`:

```
L_mono = Σ max(0, φ(t_i_depth) - φ(t_j_depth) + margin)
```
where `φ = nn.Linear(hidden_size // 2, 1)` (trainable scalar readout), `margin=1.0`.

### 4.6 LoRA Configuration (Stage 2 only)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
```

Trainable LoRA parameters: ~25M (small relative to 7B base).

---

## 5. Training

### 5.1 Stage 1 — Warm up Proj (frozen base model)

**Goal**: Train Proj to produce planning vectors that (a) contain enough information to reconstruct the next reasoning step (via AuxDec) and (b) encode hierarchical structure (via structural losses).

**Trainable modules**: Proj, AuxDec, structural heads.
**Frozen**: entire base model (including lm_head).

**Forward pass**:
1. Run base model on full sequence with `output_hidden_states=True`.
2. Extract `h_plan_i` = last hidden state at each `[PLAN]` position.
3. Compute `t_i = Proj(h_plan_i)`.
4. Add `t_i` to the hidden state at each `[PLAN]` position: `modified_h = h_plan_i + t_i`.
5. Run `lm_head` on the full modified last hidden layer to get logits.
6. Feed `t_i` and corresponding step token embeddings/ids into AuxDec to compute reconstruction loss.
7. Feed `t_i` slices into structural heads.

**Loss**:
```
L_stage1 = L_lm + λ_aux * L_auxdec + λ_seg * L_structural_seg + λ_depth * L_structural_depth
```

- **`L_lm`**: standard causal LM next-token prediction loss, computed ONLY on step tokens (question tokens and `[PLAN]` tokens are masked out). Gradients flow through `lm_head → modified hidden states → t_i → Proj`. The `lm_head` is frozen; its weights are not updated, but gradients pass through it to reach `t_i`.
- **`L_auxdec`**: cross-entropy loss from AuxDec reconstructing each step's tokens from `t_i`.
- **`L_structural_seg`**: segment loss (CE in simple mode, InfoNCE in contrastive mode).
- **`L_structural_depth`**: depth loss (MSE in simple mode, margin hinge in contrastive mode).
- Defaults: `λ_aux = 1.0`, `λ_seg = 0.1`, `λ_depth = 0.1`. Configurable via args.

**Training config**:
```yaml
learning_rate: 2e-4
weight_decay: 0.01
warmup_ratio: 0.05
num_train_epochs: 3
max_grad_norm: 1.0
lr_scheduler_type: cosine
optimizer: AdamW
```

### 5.2 Stage 2 — Joint fine-tuning (LoRA + Proj)

**Goal**: Teach the base model to (a) emit `[PLAN]` tokens at step boundaries, (b) produce useful hidden states at `[PLAN]` positions for Proj, and (c) consume the injected planning vectors when generating subsequent step tokens.

**Trainable modules**: LoRA adapters on base model, Proj (continued from Stage 1), `[PLAN]` token embedding.
**Frozen**: base model weights (except LoRA), lm_head.
**Discarded from Stage 1**: AuxDec, structural heads (not used in Stage 2).

**Two-pass forward (handles circular dependency)**:

The planning vector `t_i` depends on `h_i` (the hidden state at `[PLAN]`), but `t_i` should be injected during the forward pass that produces `h_i`. We break this circularity with a two-pass approach:

```
Pass 1 (no plan injection):
    Run full forward pass on the sequence as-is (no planning vectors injected).
    Collect h_i = hidden_state at each [PLAN] position.
    Compute t_i = Proj(stop_gradient(h_i)) for each plan position.

Pass 2 (with plan injection):
    At each position immediately after [PLAN], add t_i to the input embedding:
        input_{k+1} = Embed(r_{k+1}) + t_i
    Run full forward pass with these modified embeddings.
    Compute logits from this pass.
```

**Why `stop_gradient(h_i)`**: Prevents backpropagation through the circular dependency (Pass 2 → t_i → Pass 1). Gradients flow only through Pass 2. As training progresses, the two passes converge — the model adapts to produce hidden states consistent with the plan vectors it receives.

**Loss**:
```
L_stage2 = L_lm  (standard causal LM loss over full sequence from Pass 2)
```

The loss is computed on ALL tokens including `[PLAN]` tokens (so the model learns to predict when `[PLAN]` should appear) and step tokens (so the model learns to use injected plans). Question tokens are still masked.

**Training config**:
```yaml
learning_rate: 1e-4  # lower than Stage 1 since we're tuning the base model
weight_decay: 0.01
warmup_ratio: 0.05
num_train_epochs: 2
max_grad_norm: 1.0
lr_scheduler_type: cosine
optimizer: AdamW
```

**Memory note**: The two-pass forward doubles compute per batch. With LoRA (only ~25M trainable params), ZeRO-2, gradient checkpointing, and bf16, this fits within 4×A100-80G. If tight, reduce `per_device_batch_size` from 2 to 1 and increase `gradient_accumulation_steps` from 8 to 16 to keep effective batch size at 64.

---

## 6. DeepSpeed Configuration

```yaml
# Shared across both stages
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
```

---

## 7. Training Script Structure

```
src/
├── data/
│   ├── dataset.py              # PlanningTokenDataset, collate_fn
│   └── preprocessing.py        # Derive is_boundary, depth; tokenize; build step_ids
├── model/
│   ├── planning_model.py       # PlanningQwen: wraps base model + Proj, custom forward
│   │                           #   - stage1_forward(): single pass, hidden-state injection
│   │                           #   - stage2_forward(): two-pass, embedding-space injection
│   ├── proj.py                 # Linear/MLP Proj implementations
│   └── aux_decoder.py          # AuxDec (small transformer decoder for Stage 1)
├── losses/
│   ├── simple_structural.py    # CE + MSE
│   └── contrastive_structural.py  # InfoNCE + hinge
├── evaluation/
│   ├── baseline_eval.py        # CoT baseline evaluation of vanilla Qwen2.5-7B
│   ├── planning_eval.py        # Autonomous inference with [PLAN] hook
│   └── math_grading.py         # Sympy-based answer matching
├── train_stage1.py             # Stage 1 training loop (frozen base + Proj + AuxDec + structural)
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

### 8.2 Planning Token Model — Autonomous Inference

**Goal**: Evaluate the full two-stage trained model. The model autonomously emits `[PLAN]` tokens; a generation hook runs Proj and injects the planning vector.

**Autonomous inference loop** (on each GPU for its shard of test problems):

```python
# Load: base model with merged LoRA weights + trained Proj
# The model has been trained (Stage 2) to emit [PLAN] before each reasoning step.

PLAN_TOKEN_ID = tokenizer.convert_tokens_to_ids("[PLAN]")

for problem in test_shard:
    input_ids = tokenize(prompt + problem)
    generated = []
    past_key_values = None  # KV cache

    for pos in range(max_new_tokens):
        # Standard autoregressive step
        outputs = model(
            input_ids=current_token,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1)

        if next_token.item() == PLAN_TOKEN_ID:
            # === PLAN HOOK ===
            # 1. Get hidden state at [PLAN] position
            h_plan = outputs.hidden_states[-1][:, -1, :]

            # 2. Compute planning vector
            t_i = proj(h_plan)

            # 3. Inject: add t_i to the embedding of the NEXT token
            #    We don't know the next token yet, so we continue generation
            #    by modifying the input embedding at the next forward call.
            plan_vector_pending = t_i  # store for injection at next position
        else:
            plan_vector_pending = None

        generated.append(next_token.item())

        # Prepare next input
        next_embed = model.get_input_embeddings()(next_token.unsqueeze(0))
        if plan_vector_pending is not None:
            next_embed = next_embed + plan_vector_pending
            plan_vector_pending = None
        current_token_embeds = next_embed  # feed embeddings directly, not token ids

        # Check termination
        if next_token.item() == tokenizer.eos_token_id:
            break
        if contains_boxed(tokenizer.decode(generated)):
            break

    # Extract and grade answer
    answer = extract_boxed(tokenizer.decode(generated))
    correct = grade_answer(answer, ground_truth)
```

**Critical implementation detail**: When injecting `t_i`, we must feed the model input **embeddings** (not token ids) at the next position so we can add the planning vector. Most HuggingFace models support `inputs_embeds` as an alternative to `input_ids`. The implementation should:
1. Normally feed `input_ids` for efficiency.
2. When a `[PLAN]` token is detected, switch to `inputs_embeds` for the next position only, adding `t_i` to the token embedding.
3. Switch back to `input_ids` for subsequent tokens.

The KV cache remains valid throughout — we are only modifying the *input* at one position, not rewriting past states.

**Use the same prompt template and generation config as baseline** for fair comparison.

**DDP**: Same sharding as baseline.

### 8.3 Planning Token Model — External Controller (Ablation)

**Goal**: Ablation that evaluates the Stage 1 Proj alone (no LoRA, no autonomous planning). Uses an external controller to forcibly inject `[PLAN]` at step boundaries, identical to the original pipeline spec. This measures the value of Proj in isolation.

**External controller inference loop** (on each GPU for its shard of test problems):

```python
# Load: vanilla Qwen2.5-7B (no LoRA) + trained Proj from Stage 1 checkpoint

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

**Use the same prompt template and generation config as baseline** for fair comparison.

### 8.4 Metrics

Report for baseline, external-controller (Stage 1 ablation), and autonomous (full pipeline):
- **Overall accuracy** (% of 500 problems correct)
- **Accuracy by difficulty level** (Level 1–5)
- **Accuracy by subject** (7 subjects)
- **Average steps generated** (for planning models)
- **Average `[PLAN]` tokens emitted** (for autonomous model — should be close to number of reasoning steps)

---

## 9. Key Implementation Notes

1. **Stage 1 frozen model**: After loading Qwen2.5-7B, call `model.requires_grad_(False)`. Only Proj, AuxDec, and structural heads should be in the optimizer's param groups. Verify with a unit test that `Proj.weight.grad` is non-None after `L_lm.backward()` (gradients flow through the frozen `lm_head` to reach `t_i`).

2. **Stage 2 LoRA setup**: Load the base model, apply LoRA via `peft`, then load the trained Proj checkpoint from Stage 1. The optimizer param groups should include LoRA parameters, Proj parameters, and the `[PLAN]` token embedding. Everything else is frozen.

3. **Stage 2 two-pass memory**: Two full forward passes per batch approximately doubles compute. To fit in memory: use gradient checkpointing, bf16, and if needed reduce micro-batch size to 1 (with gradient_accumulation_steps=16 to maintain effective batch size of 64). Pass 1 runs under `torch.no_grad()` — only Pass 2 needs gradients.

4. **Collation**: Sequences in a batch will have different numbers of steps (and thus different numbers of `[PLAN]` tokens). Pad sequences to max length in batch. The `step_ids` array handles masking. The structural losses (Stage 1) should gather planning vectors across the batch and compute structure losses on each solution independently (segment/depth labels are per-solution, not cross-solution).

5. **AuxDec data feeding**: For each `[PLAN]` position `i`, AuxDec receives `t_i` and the token embeddings + ids of step `i`. Since steps have variable length, either pad step tokens to a max step length (e.g., 256) or process steps sequentially within each batch element. The former is simpler and recommended.

6. **Contrastive loss batching**: InfoNCE requires multiple planning vectors. Within a single sequence, there may only be 3–5 planning tokens, giving few contrastive pairs. Gather planning vectors across all sequences in the micro-batch to form a larger pool. Positives = same segment within the same solution. Negatives = everything else. Accumulate across gradient accumulation steps if needed, or just use the micro-batch.

7. **Stage 2 `[PLAN]` prediction**: In Stage 2 the LM loss includes `[PLAN]` token positions. The model must learn to predict `[PLAN]` after the last token of each step. Since `[PLAN]` is a new token only seen at step boundaries, the LoRA adapters and the `[PLAN]` embedding are the degrees of freedom that enable this.

8. **Checkpoint management**: Save Stage 1 outputs as `{proj.pt, aux_heads.pt}`. Stage 2 loads `proj.pt` and ignores `aux_heads.pt`. Save Stage 2 outputs as `{lora_adapters/, proj.pt}`. For inference, merge LoRA into base model weights using `model.merge_and_unload()` and load `proj.pt` separately.

---

## 10. Command-Line Interface

```bash
# Stage 1: Warm up Proj
deepspeed --num_gpus=4 src/train_stage1.py \
    --data_path data/prm800k_annotated.jsonl \
    --model_name Qwen/Qwen2.5-7B \
    --proj_type mlp \
    --structural_loss simple \
    --lambda_aux 1.0 \
    --lambda_seg 0.1 \
    --lambda_depth 0.1 \
    --auxdec_layers 2 \
    --auxdec_heads 8 \
    --max_seq_len 2048 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --lr 2e-4 \
    --output_dir checkpoints/stage1 \
    --deepspeed configs/deepspeed_config.json

# Stage 2: Joint fine-tuning with LoRA
deepspeed --num_gpus=4 src/train_stage2.py \
    --data_path data/prm800k_annotated.jsonl \
    --model_name Qwen/Qwen2.5-7B \
    --proj_checkpoint checkpoints/stage1/proj.pt \
    --proj_type mlp \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj v_proj \
    --lora_dropout 0.05 \
    --max_seq_len 2048 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 2 \
    --lr 1e-4 \
    --output_dir checkpoints/stage2 \
    --deepspeed configs/deepspeed_config.json

# Baseline evaluation
torchrun --nproc_per_node=4 src/evaluation/baseline_eval.py \
    --model_name Qwen/Qwen2.5-7B \
    --max_new_tokens 1024 \
    --output_file results/baseline.json

# Stage 1 ablation: external controller
torchrun --nproc_per_node=4 src/evaluation/planning_eval.py \
    --model_name Qwen/Qwen2.5-7B \
    --proj_checkpoint checkpoints/stage1/proj.pt \
    --proj_type mlp \
    --inference_mode external_controller \
    --max_steps 20 \
    --max_step_tokens 256 \
    --output_file results/stage1_external.json

# Full pipeline: autonomous inference
torchrun --nproc_per_node=4 src/evaluation/planning_eval.py \
    --model_name Qwen/Qwen2.5-7B \
    --lora_checkpoint checkpoints/stage2/lora_adapters \
    --proj_checkpoint checkpoints/stage2/proj.pt \
    --proj_type mlp \
    --inference_mode autonomous \
    --max_new_tokens 1024 \
    --output_file results/stage2_autonomous.json
```
