"""HypPlanModel: wraps base LLM + ProjMLP for planning vector insertion."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.projections import ProjMLP


class HypPlanModel(nn.Module):
    """Wraps base LLM + Proj for Stage 1 training.

    Planning vectors z (same dim as hidden states) are inserted into the
    embedding sequence as virtual tokens before each reasoning step.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_name = config["model"]["base_model"]
        proj_hidden_dims = config["model"]["proj_hidden_dims"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base LLM
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        hidden_dim = self.base_model.config.hidden_size

        # Projection: hidden_dim -> hidden_dim (tangent vector usable as virtual token)
        target_norm = config["model"].get("plan_vector_scale", 1.0)
        self.proj = ProjMLP(hidden_dim, proj_hidden_dims, target_norm=target_norm)

    def freeze_base_model(self):
        """Freeze all base LLM parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def get_hidden_states(self, input_ids, attention_mask=None):
        """Run base LLM and return last hidden states (no gradient)."""
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
            boundary_positions: (B, num_steps) positions of step boundaries (-1 = padding).
        Returns:
            t: (B, num_steps, H+1) hyperboloid points.
            z: (B, num_steps, H) tangent vectors (same dim as embeddings).
            valid_mask: (B, num_steps) bool mask.
        """
        valid_mask = boundary_positions >= 0
        safe_positions = boundary_positions.clamp(min=0)
        safe_positions_expanded = safe_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        h = torch.gather(hidden_states, 1, safe_positions_expanded)
        t, z = self.proj(h)
        return t, z, valid_mask

    def forward_stage1(self, input_ids, attention_mask,
                       boundary_positions, grad_accum_denom: float = 1.0):
        """Stage 1 per-step training loss.

        For each sample and each valid step i, builds the truncated sequence
        [x, r_{<i}, t_i, r_i] as embeddings, forwards through the frozen LLM,
        and computes cross-entropy on r_i token positions. The loss is scaled
        by 1/grad_accum_denom and backward is called immediately so that the
        computation graph is freed between steps (memory-bounded).

        Step token ranges are derived from boundary_positions:
          r_i = input_ids[bp[i]+1 : bp[i+1]+1]  for i < K-1
          r_i = input_ids[bp[i]+1 : real_len]    for i == K-1 (last step)

        Only ProjMLP parameters accumulate gradients (base LLM is frozen).

        Args:
            input_ids: (B, L) original token IDs.
            attention_mask: (B, L)
            boundary_positions: (B, num_steps) with -1 padding. boundary_positions[b, i]
                is the last token index before step i's content starts.
            grad_accum_denom: scale factor applied before backward.

        Returns:
            avg_loss: float — mean of per-step scalar losses.
            z_norm_mean: float — mean L2 norm of ProjMLP outputs across all steps.
        """
        # Pass 1: hidden states from full sequence (no grad)
        hidden_states = self.get_hidden_states(input_ids, attention_mask)  # (B, L, H)

        embed_table = self.base_model.get_input_embeddings()
        total_loss = 0.0
        total_steps = 0
        z_norms: list[float] = []

        B = input_ids.size(0)
        for b in range(B):
            valid_bp = boundary_positions[b][boundary_positions[b] >= 0]
            K = valid_bp.size(0)
            real_len = attention_mask[b].sum().item()

            for i in range(K):
                bpos = valid_bp[i].item()

                # Derive r_i token range from boundary positions
                step_start = bpos + 1
                step_end = valid_bp[i + 1].item() + 1 if i < K - 1 else real_len
                if step_start >= step_end:
                    continue
                step_ids = input_ids[b, step_start:step_end]     # (|r_i|,)

                # Recompute t_i with a fresh graph so backward() below doesn't
                # invalidate earlier or later iterations.
                h_i = hidden_states[b, bpos].unsqueeze(0)        # (1, H), detached
                _, z_i = self.proj(h_i)                          # (1, H), grad-enabled
                z_norms.append(z_i.detach().float().norm().item())

                # Build [x, r_{<i}, t_i, r_i] as inputs_embeds
                prefix_ids = input_ids[b, : bpos + 1]
                prefix_embeds = embed_table(prefix_ids)          # (prefix_len, H)
                step_embeds = embed_table(step_ids)              # (|r_i|, H)
                prefix_len = prefix_embeds.size(0)
                s_len = step_embeds.size(0)

                full_embeds = torch.cat(
                    [prefix_embeds, z_i, step_embeds], dim=0
                ).unsqueeze(0)                                    # (1, L_i, H)
                L_i = full_embeds.size(1)

                # Labels: predict r_i tokens after t_i
                full_labels = torch.full(
                    (1, L_i), -100, dtype=torch.long, device=input_ids.device
                )
                full_labels[0, prefix_len + 1 : prefix_len + 1 + s_len] = step_ids

                outputs = self.base_model(inputs_embeds=full_embeds)
                logits = outputs.logits                           # (1, L_i, V)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = full_labels[:, 1:].contiguous()
                loss_i = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # Backward immediately — frees the graph so we can iterate.
                (loss_i / grad_accum_denom).backward()

                total_loss += loss_i.item()
                total_steps += 1

        avg_loss = total_loss / total_steps
        z_norm_mean = sum(z_norms) / len(z_norms)
        return avg_loss, z_norm_mean
