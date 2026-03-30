"""HypPlanModel: wraps base LLM + ProjMLP + [PLAN] token management."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.proj import ProjMLP, ProjectBack


class HypPlanModel(nn.Module):
    """Wraps base LLM + Proj + [PLAN] token and embedding injection."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_name = config["model"]["base_model"]
        hyp_dim = config["model"]["hyp_dim"]
        proj_hidden_dims = config["model"]["proj_hidden_dims"]

        # Load tokenizer and add [PLAN] token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        plan_token = config["model"]["plan_token"]
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": [plan_token]})
        self.plan_token_id = self.tokenizer.convert_tokens_to_ids(plan_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base LLM to CPU; caller moves to the correct GPU via .to(device)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        hidden_dim = self.base_model.config.hidden_size

        # Proj MLP
        self.proj = ProjMLP(hidden_dim, hyp_dim, proj_hidden_dims)

        # Project back from tangent space to embedding space
        embed_dim = self.base_model.config.hidden_size
        self.project_back = ProjectBack(hyp_dim, embed_dim)

    def freeze_base_model(self):
        """Freeze all base LLM parameters (Stages 1 & 2)."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def freeze_plan_token_embedding(self):
        """Freeze the [PLAN] token embedding (Stage 3 spec)."""
        embed_weight = self.base_model.get_input_embeddings().weight
        # Register a hook to zero out the gradient for the [PLAN] token row
        self._plan_embed_hook = embed_weight.register_hook(
            lambda grad: self._zero_plan_grad(grad)
        )

    def _zero_plan_grad(self, grad):
        grad = grad.clone()
        grad[self.plan_token_id] = 0
        return grad

    def get_hidden_states(self, input_ids, attention_mask=None):
        """Run base LLM and return last hidden states (no gradient through LLM).

        Args:
            input_ids: (B, L) token IDs.
            attention_mask: (B, L) attention mask.
        Returns:
            (B, L, H) last layer hidden states.
        """
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return outputs.hidden_states[-1]

    def compute_plan_vectors(self, hidden_states, boundary_positions):
        """Extract hidden states at step boundaries and compute planning vectors.

        Args:
            hidden_states: (B, L, H) from base LLM.
            boundary_positions: (B, num_steps) positions of step boundaries.
                Padding value -1 means no step at that index.
        Returns:
            t_list: (B, num_steps, hyp_dim+1) hyperboloid points.
            z_list: (B, num_steps, hyp_dim) tangent vectors.
            valid_mask: (B, num_steps) bool mask for valid steps.
        """
        B, num_steps = boundary_positions.shape
        device = hidden_states.device
        valid_mask = boundary_positions >= 0

        # Clamp to valid indices for gathering
        safe_positions = boundary_positions.clamp(min=0)  # (B, num_steps)
        safe_positions_expanded = safe_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        h = torch.gather(hidden_states, 1, safe_positions_expanded)  # (B, num_steps, H)

        # Compute planning vectors
        t, z = self.proj(h)  # (B, num_steps, hyp_dim+1), (B, num_steps, hyp_dim)
        return t, z, valid_mask

    def inject_plan_vectors(self, input_ids, z_list, inject_positions, valid_mask,
                            attention_mask=None):
        """Build modified embeddings with planning vectors injected and run LLM.

        Args:
            input_ids: (B, L) token IDs.
            z_list: (B, num_steps, hyp_dim) tangent vectors.
            inject_positions: (B, num_steps) positions to inject at.
            valid_mask: (B, num_steps) which steps are valid.
            attention_mask: (B, L) attention mask.
        Returns:
            CausalLMOutput with logits.
        """
        # Get base embeddings
        embeddings = self.base_model.get_input_embeddings()(input_ids)  # (B, L, E)
        embeddings = embeddings.clone()  # avoid in-place modification

        # Project tangent vectors back to embedding space
        delta = self.project_back(z_list)  # (B, num_steps, E)

        # Inject at specified positions
        B = input_ids.size(0)
        for b in range(B):
            for s in range(inject_positions.size(1)):
                if valid_mask[b, s]:
                    pos = inject_positions[b, s].item()
                    if 0 <= pos < embeddings.size(1):
                        embeddings[b, pos] = embeddings[b, pos] + delta[b, s]

        # Forward with modified embeddings
        outputs = self.base_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs

    def forward_stage1(self, input_ids, attention_mask, labels,
                       boundary_positions, inject_positions):
        """Stage 1 forward: frozen LLM, train Proj only.

        Returns:
            loss: scalar LM loss on step tokens.
            t: (B, num_steps, hyp_dim+1) hyperboloid points.
        """
        # Pass 1: collect hidden states (no grad through LLM)
        hidden_states = self.get_hidden_states(input_ids, attention_mask)

        # Compute planning vectors
        t, z, valid_mask = self.compute_plan_vectors(hidden_states, boundary_positions)

        # Pass 2: inject and compute loss
        outputs = self.inject_plan_vectors(
            input_ids, z, inject_positions, valid_mask, attention_mask
        )

        # Compute LM loss on labeled positions only
        loss = self._compute_lm_loss(outputs.logits, labels)
        return loss, t

    def forward_stage2(self, input_ids, attention_mask, labels,
                       boundary_positions, inject_positions):
        """Stage 2 forward: same as Stage 1, returns planning vectors for tree loss."""
        return self.forward_stage1(
            input_ids, attention_mask, labels,
            boundary_positions, inject_positions,
        )

    def forward_stage3_pass1(self, input_ids, attention_mask, plan_positions):
        """Stage 3 pass 1: run LLM with LoRA, collect hidden states at [PLAN] positions.

        Args:
            input_ids: (B, L) includes [PLAN] tokens.
            attention_mask: (B, L).
            plan_positions: (B, num_plans) positions of [PLAN] tokens.
        Returns:
            t: (B, num_plans, hyp_dim+1) hyperboloid points.
            z: (B, num_plans, hyp_dim) tangent vectors.
            valid_mask: (B, num_plans) bool mask.
        """
        # Run full sequence through LLM (with LoRA) — need hidden states
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        # Collect hidden states at the token BEFORE each [PLAN]
        # plan_positions are the [PLAN] token positions; we want position-1
        before_plan = (plan_positions - 1).clamp(min=0)
        valid_mask = plan_positions >= 0

        safe_pos = before_plan.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        h = torch.gather(hidden_states, 1, safe_pos)

        # stop_gradient on h (as per spec)
        t, z = self.proj(h.detach())
        return t, z, valid_mask

    def forward_stage3_pass2(self, input_ids, attention_mask, labels,
                             z_list, inject_positions, valid_mask):
        """Stage 3 pass 2: inject planning vectors and compute full LM loss.

        Args:
            input_ids: (B, L) includes [PLAN] tokens.
            attention_mask: (B, L).
            labels: (B, L) with -100 for masked positions.
            z_list: (B, num_plans, hyp_dim) tangent vectors from pass 1.
            inject_positions: (B, num_plans) positions right after [PLAN].
            valid_mask: (B, num_plans) which plans are valid.
        Returns:
            loss: scalar LM loss.
        """
        outputs = self.inject_plan_vectors(
            input_ids, z_list, inject_positions, valid_mask, attention_mask
        )
        loss = self._compute_lm_loss(outputs.logits, labels)
        return loss

    def _compute_lm_loss(self, logits, labels):
        """Standard causal LM loss with label masking."""
        # Shift: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss
