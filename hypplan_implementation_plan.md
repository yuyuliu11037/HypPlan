# HypPlan: Hyperbolic Planning Tokens for LLM Reasoning — Implementation Plan

## 0. Project Setup

**Location**: `~/HypPlan` (from scratch)
**Hardware**: 8× NVIDIA A6000 (48 GB each)
**Base LLM**: `Qwen/Qwen2.5-7B` (base, not instruct)

### 0.1 Directory Structure

```
~/HypPlan/
├── configs/
│   └── default.yaml           # all hyperparameters in one place
├── data/
│   ├── math_filtered.jsonl    # (already exists) raw processed data
│   └── reasoning_trees.jsonl  # (already exists) prebuilt tree structures
├── src/
│   ├── data/
│   │   ├── dataset_stage1.py  # Stage 1 & 3 dataset (correct generations)
│   │   ├── dataset_stage2.py  # Stage 2 dataset (all 16 generations + tree)
│   │   └── utils.py           # step splitting, answer extraction
│   ├── model/
│   │   ├── proj.py            # Proj MLP + Lorentz ops
│   │   ├── hyperbolic.py      # Lorentz manifold utilities
│   │   ├── plan_model.py      # wrapper: base LLM + Proj + [PLAN] token
│   │   └── lora_utils.py      # LoRA setup for Stage 3
│   ├── training/
│   │   ├── stage1.py          # Stage 1 training loop
│   │   ├── stage2.py          # Stage 2 training loop
│   │   └── stage3.py          # Stage 3 training loop
│   ├── inference/
│   │   └── generate.py        # inference with [PLAN] hook
│   └── eval/
│       └── evaluate.py        # \boxed{} extraction + accuracy
├── scripts/
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── run_stage3.sh
│   └── run_eval.sh
└── requirements.txt
```

### 0.2 Dependencies

```
torch >= 2.1
transformers >= 4.37
peft (for LoRA)
accelerate
deepspeed
geoopt (Lorentz/hyperbolic ops — use as reference, but implement core ops manually for control)
wandb
pyyaml
```

### 0.3 Config File (`configs/default.yaml`)

Define all hyperparameters here. Key ones:

```yaml
model:
  base_model: "Qwen/Qwen2.5-7B"
  hyp_dim: 64               # hyperbolic embedding dim (hyperparameter)
  proj_hidden_dims: [2048]   # Proj MLP hidden layers (hyperparameter)
  plan_token: "[PLAN]"

data:
  math_filtered: "data/math_filtered.jsonl"
  reasoning_trees: "data/reasoning_trees.jsonl"
  step_delimiter: "\n\n"
  max_seq_len: 2048

stage1:
  lr: 1e-4
  epochs: 3
  batch_size: 4              # per GPU
  grad_accum: 4

stage2:
  lr: 1e-4
  epochs: 3
  batch_size: 4
  grad_accum: 4
  tree_loss_weight: 1.0      # weight of L_tree relative to L_plan
  c_init: 1.0                # learnable scaling factor initial value

stage3:
  lr: 2e-5
  epochs: 3
  batch_size: 2              # smaller due to two-pass
  grad_accum: 8
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

---

## 1. Data Preparation (`src/data/`)

### 1.1 Step Splitting Utility (`utils.py`)

```
Input:  a generation string (one of the 16 per problem)
Output: list of step strings

Logic:
  1. Split on "\n\n"
  2. Strip each segment
  3. Discard empty segments
```

Also implement answer extraction:
```
Input:  a generation string
Output: the content inside the last \boxed{...}

Logic:
  1. Find the last occurrence of \boxed{
  2. Handle nested braces to extract the full content
  3. Return None if no \boxed{} found
```

### 1.2 Stage 1 & 3 Dataset (`dataset_stage1.py`)

**Source**: `math_filtered.jsonl`
**Filter**: only correct generations (use the `correct` field as a boolean mask over `generations`)

Each sample is one correct generation for one problem:
```python
{
    "problem": str,              # the problem text
    "steps": list[str],          # generation split by \n\n
    "full_generation": str,      # the raw generation string
}
```

**Tokenization**:
- Tokenize problem as prefix
- Tokenize each step separately to know step boundary token positions
- Record the token index of the last token of each step (these are where we extract h_i)
- For Stage 3: insert the [PLAN] token ID before each step's tokens

### 1.3 Stage 2 Dataset (`dataset_stage2.py`)

**Source**: `reasoning_trees.jsonl`
**Uses all 16 generations** (not just correct ones)

Each sample is one generation for one problem, but with tree metadata:
```python
{
    "problem": str,
    "steps": list[str],           # this generation's steps
    "tree_node_ids": list[int],   # which tree node each step maps to
    "pairwise_distances": list[list[float]],  # full tree distance matrix (for the whole problem)
}
```

**Mapping steps to tree nodes**:
- For a given generation `g_id`, walk the tree: at each depth `d`, find the node at depth `d` whose `generation_ids` list contains `g_id`. That node's `node_id` is the tree node for step `d`.
- The `generation_leaf_ids` field tells you which leaf node each generation ends at; use this to verify the mapping.

**Tree distance lookup**:
- `pairwise_distances` is pre-computed in the data. Index it by `tree_node_ids` to get `d_tree(s_i, s_j)` for any pair of steps within a problem.

---

## 2. Hyperbolic Utilities (`src/model/hyperbolic.py`)

Implement the **Lorentz (hyperboloid) model** of hyperbolic space.

### 2.1 Core Operations

```
A point on the hyperboloid in R^{d+1}: x = (x_0, x_1, ..., x_d) where x_0 > 0 and -x_0^2 + x_1^2 + ... + x_d^2 = -1

Minkowski inner product:
  <x, y>_L = -x_0*y_0 + sum(x_i*y_i for i in 1..d)

Distance:
  d_H(x, y) = arccosh(-<x, y>_L)

Exponential map at origin o = (1, 0, ..., 0):
  Input: v in R^d (tangent vector)
  v_norm = ||v||_2
  exp_o(v) = (cosh(v_norm), sinh(v_norm) * v / v_norm)
  Handle v_norm ≈ 0 with Taylor expansion for numerical stability

Origin:
  o = (1, 0, 0, ..., 0) in R^{d+1}
```

### 2.2 Numerical Stability

- Clamp `arccosh` input to `[1 + eps, ...]` to avoid NaN
- Use `torch.clamp` on norms before division
- For very small norms in `exp_o`, use first-order Taylor: `exp_o(v) ≈ (1, v)`

---

## 3. Model Components (`src/model/`)

### 3.1 Proj MLP (`proj.py`)

```python
class ProjMLP(nn.Module):
    """Maps LLM hidden states to Lorentz hyperbolic space."""
    
    def __init__(self, hidden_dim, hyp_dim, proj_hidden_dims):
        # hidden_dim: LLM hidden size (e.g. 4096 for Qwen2.5-7B)
        # hyp_dim: dimension of tangent vector (so output of MLP is R^hyp_dim)
        # proj_hidden_dims: list of hidden layer sizes, e.g. [2048]
        
        # Build MLP: hidden_dim -> proj_hidden_dims[0] -> ... -> hyp_dim
        # Use GELU activation between layers
        # Final output is a tangent vector in R^hyp_dim
    
    def forward(self, h):
        z = self.mlp(h)         # R^hyp_dim, tangent vector
        t = exp_o(z)            # R^{hyp_dim+1}, point on hyperboloid
        return t, z             # return both for flexibility
```

### 3.2 Plan Model Wrapper (`plan_model.py`)

```python
class HypPlanModel(nn.Module):
    """Wraps base LLM + Proj. Manages [PLAN] token and embedding injection."""
    
    def __init__(self, base_model, proj, plan_token_id):
        self.base_model = base_model  # Qwen2.5-7B
        self.proj = proj
        self.plan_token_id = plan_token_id
    
    # Add [PLAN] to tokenizer and resize embeddings.
    # IMPORTANT: freeze the [PLAN] embedding after random init (Stage 3 spec).
```

---

## 4. Training Stages (`src/training/`)

### 4.1 Stage 1: Warm Up Proj

**Goal**: Teach `Proj` to produce planning vectors that help the LLM predict the next step.

**What is frozen**: entire base LLM `f`
**What is trainable**: `Proj` only

**Forward pass** (for one sample with steps `[r_0, r_1, ..., r_n]`):

```
For each step boundary i:
  1. Run frozen LLM on [problem, r_0, r_1, ..., r_{i-1}] → get hidden state h_i
     (h_i = hidden state at the last token position before step i starts)
  2. t_i = Proj(h_i)  →  point on hyperboloid
  3. Inject t_i by adding it to the embedding of the first token of step r_i
     (similar to Stage 3: input_embedding = Embed(first_token_of_step_i) + linear_proj(t_i))
     NOTE: you need a small linear layer to project from hyp_dim+1 back to embedding_dim.
     Alternatively, use only the tangent vector z_i (R^hyp_dim) and project that.
  4. Continue LLM forward pass over step r_i tokens, compute cross-entropy loss
     on predicting tokens of r_i
```

**Loss**: `L_plan = - sum_i log p(r_i | problem, r_{<i}, t_i)`
This is standard causal LM loss over step tokens, but with the planning vector injected.

**Implementation detail**: Since the LLM is frozen, you can:
- Run the full sequence through the frozen LLM once to collect all hidden states at step boundaries
- Then for each step, inject `t_i` and re-run only that step's tokens (or the full sequence with injections). The simpler approach: run the full sequence with `t_i` injections all at once, masking the loss to only count step tokens.

**Distributed**: Use DeepSpeed ZeRO-2 or FSDP across 8 GPUs. Since only Proj is trainable, memory should be manageable.

### 4.2 Stage 2: Structurize Proj with Tree Loss

**Goal**: Make the hyperbolic planning vectors respect the tree structure of reasoning.

**What is frozen**: entire base LLM `f`
**What is trainable**: `Proj`, learnable scalar `c`

**Data**: all 16 generations per problem, each mapped to tree nodes.

**Forward pass** (same as Stage 1 — compute h_i at step boundaries, then t_i = Proj(h_i))

**Loss**: `L = L_plan + λ * L_tree`

```
L_tree for one problem:
  Collect all (t_i, node_id_i) pairs across all 16 generations for this problem.
  For each pair (i, j) of steps (can sample pairs for efficiency):
    L_tree += (d_H(t_i, t_j) - c * d_tree(node_id_i, node_id_j))^2
  
  d_tree is looked up from pairwise_distances using node_ids.
  c is a learnable scalar (initialized to c_init from config).
```

**Sampling strategy for pairs**: Computing all pairs is O(n^2) per problem which may be expensive. Sample a fixed number of pairs per problem (e.g., 256 pairs). Include both same-node pairs (should have small d_H) and different-branch pairs (should have large d_H).

**Batching concern**: Different problems have different tree structures. Within a single batch, L_tree is computed per-problem. L_plan is computed as in Stage 1.

### 4.3 Stage 3: Joint Fine-Tuning with LoRA

**Goal**: Train the LLM (via LoRA) to use injected planning vectors and to emit `[PLAN]` tokens.

**What is trainable**: LoRA adapters on `f`, `Proj` (continued)
**What is frozen**: base LLM weights, `lm_head`, `[PLAN]` token embedding

**Data**: correct generations with `[PLAN]` token inserted before each step.

**Forward pass** (two-pass):

```
Pass 1 (no gradient through LLM for hidden state collection):
  1. Tokenize: [problem tokens] [step_0 tokens] [PLAN] [step_1 tokens] [PLAN] [step_2 tokens] ...
  2. Run full sequence through LLM normally (with LoRA)
  3. At each [PLAN] position, collect hidden state h_i from the token just before [PLAN]
     (i.e., the last token of step_{i-1}, or the last problem token for the first [PLAN])
  4. Compute t_i = Proj(stop_gradient(h_i))  ← stop gradient on h_i

Pass 2 (compute loss):
  1. Modify input embeddings: at each position right after [PLAN],
     add t_i to that token's embedding:
       modified_embed[pos_after_plan_i] = Embed(token_at_that_pos) + project_back(t_i)
  2. Run LLM again with modified embeddings → get logits
  3. Compute standard causal LM loss over ALL tokens
     (the model must learn to predict [PLAN] tokens AND step tokens)
```

**Memory concern**: Two forward passes on 7B model. With LoRA (r=16), gradient checkpointing, and DeepSpeed ZeRO-3, this should fit on 8× A6000 with batch_size=2 per GPU and grad_accum=8.

**CRITICAL**: The `[PLAN]` token embedding is randomly initialized and FROZEN. The model learns to predict `[PLAN]` via the `lm_head` (which is also frozen per spec). This means the LoRA adapters must learn representations that make `[PLAN]` predictable from context. Verify this works — if the model never learns to emit `[PLAN]`, you may need to unfreeze `lm_head` or use a different strategy.

**Potential issue to watch**: `lm_head` is frozen and `[PLAN]` embedding is random. The model needs to learn (via LoRA) to produce hidden states that, when projected by the frozen `lm_head`, give high probability to `[PLAN]`. Since LoRA modifies the hidden states, this should be possible in principle, but monitor [PLAN] prediction accuracy during training.

---

## 5. Inference (`src/inference/generate.py`)

Standard autoregressive generation with a hook:

```python
def generate(model, tokenizer, problem, max_new_tokens=1024):
    input_ids = tokenizer.encode(problem)
    
    for step in range(max_new_tokens):
        logits, hidden = model.forward(input_ids)  # need hidden states
        next_token = sample_or_greedy(logits[:, -1])
        
        if next_token == plan_token_id:
            # Hook: compute planning vector
            h_i = hidden[:, -1, :]          # hidden state at [PLAN] position
            t_i = proj(h_i)                 # project to hyperboloid
            
            # Append [PLAN] to sequence
            input_ids = append(input_ids, next_token)
            
            # For the NEXT token: its embedding will have t_i added
            # Store t_i to inject when processing the next position
            pending_plan_vector = t_i
        else:
            if pending_plan_vector is not None:
                # Inject t_i into this token's embedding
                # (handle this in the model's embedding layer)
                pass
            input_ids = append(input_ids, next_token)
            pending_plan_vector = None
        
        if next_token == eos_token_id:
            break
    
    return tokenizer.decode(input_ids)
```

**Implementation note**: You'll need to modify the model's forward pass to accept an optional `plan_injection` dict mapping positions to plan vectors. Or implement a custom `generate` function with KV-cache that handles the hook.

---

## 6. Evaluation (`src/eval/evaluate.py`)

### 6.1 Answer Extraction

```
Extract content from the last \boxed{...} in the generated text.
Handle nested braces: track brace depth, find matching closing brace.
Normalize: strip whitespace, handle LaTeX formatting (e.g., \frac{1}{2} = 0.5).
```

### 6.2 Metrics

- **Accuracy**: exact match of extracted answer vs ground truth answer (from `solution` field)
- **Pass@1**: accuracy on a single greedy generation
- Report accuracy broken down by `level` (1-5) and `type`

### 6.3 Baselines

- **CoT-SFT**: Fine-tune Qwen2.5-7B with LoRA on the same correct generations (same data, same compute budget), standard causal LM loss, no planning tokens. This is the primary baseline.

---

## 7. Execution Order

```
Step 1:  Set up project structure, install dependencies
Step 2:  Implement hyperbolic.py (Lorentz ops) — unit test with known distances
Step 3:  Implement data loading (utils.py, dataset_stage1.py, dataset_stage2.py)
         Verify: load math_filtered.jsonl, check step splitting, check tree node mapping
Step 4:  Implement proj.py and plan_model.py
Step 5:  Implement Stage 1 training loop — train Proj
         Checkpoint: Proj weights saved
Step 6:  Implement Stage 2 training loop — add tree loss
         Checkpoint: Proj weights updated, learnable c saved
Step 7:  Implement Stage 3 training loop — LoRA + two-pass
         Checkpoint: LoRA adapters + Proj weights saved
Step 8:  Implement inference with [PLAN] hook
Step 9:  Implement evaluation pipeline
Step 10: Train CoT-SFT baseline (standard LoRA fine-tune, same data)
Step 11: Run evaluation on both, compare
```

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `[PLAN]` never emitted at inference (frozen lm_head + random embedding) | Monitor [PLAN] prediction loss during Stage 3. If it doesn't decrease, try unfreezing lm_head or initializing [PLAN] embedding as average of punctuation tokens. |
| Stage 3 two-pass OOM on 7B model | Use gradient checkpointing, DeepSpeed ZeRO-3, reduce batch size to 1, increase grad_accum. |
| Hyperbolic distances explode/collapse | Log d_H statistics per epoch. If collapsing, increase hyp_dim. If exploding, clamp or use a smaller learning rate for Proj. |
| Tree node mapping errors | Write a unit test: for each generation, verify the mapped tree path is consistent with generation_ids at each depth. |
| L_tree dominates or is ignored | Tune tree_loss_weight λ. Start with 1.0, check gradient magnitudes of both losses. |
