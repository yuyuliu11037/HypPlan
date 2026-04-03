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

        # Load tokenizer and add [PLAN] token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        plan_token = config["model"]["plan_token"]
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": [plan_token]})
        self.plan_token_id = self.tokenizer.convert_tokens_to_ids(plan_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base LLM
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        hidden_dim = self.base_model.config.hidden_size

        # Projection: hidden_dim -> hidden_dim (tangent vector usable as virtual token)
        self.proj = ProjMLP(hidden_dim, proj_hidden_dims)

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

    def insert_plan_vectors(self, input_ids, z_list, inject_positions, valid_mask,
                            attention_mask=None, labels=None):
        """Build embeddings with planning vectors inserted as virtual tokens, then run LLM.

        For each valid step, inserts z as a new token embedding at the inject position.
        This shifts subsequent tokens right. Labels and attention mask are adjusted
        accordingly (the inserted virtual token gets label=-100 and attention=1).

        Args:
            input_ids: (B, L) token IDs.
            z_list: (B, num_steps, H) tangent vectors to insert.
            inject_positions: (B, num_steps) positions to insert before (-1 = skip).
            valid_mask: (B, num_steps) which steps are valid.
            attention_mask: (B, L) attention mask.
            labels: (B, L) labels with -100 for masked positions.
        Returns:
            logits: (B, L', vocab_size) from LLM forward.
            new_labels: (B, L') adjusted labels with -100 at inserted positions.
        """
        base_embeds = self.base_model.get_input_embeddings()(input_ids)  # (B, L, H)
        B, L, H = base_embeds.shape

        new_embeds_list = []
        new_attn_list = []
        new_labels_list = []

        for b in range(B):
            embed_b = base_embeds[b]  # (L, H)
            attn_b = attention_mask[b] if attention_mask is not None else torch.ones(L, device=embed_b.device)
            label_b = labels[b] if labels is not None else torch.full((L,), -100, dtype=torch.long, device=embed_b.device)

            # Collect valid insert positions for this sample, sorted ascending
            positions = []
            vectors = []
            for s in range(inject_positions.size(1)):
                if valid_mask[b, s]:
                    pos = inject_positions[b, s].item()
                    if 0 <= pos <= L:
                        positions.append(pos)
                        vectors.append(z_list[b, s])  # (H,)

            if not positions:
                new_embeds_list.append(embed_b)
                new_attn_list.append(attn_b)
                new_labels_list.append(label_b)
                continue

            # Sort by position (ascending), insert from back to front to keep indices stable
            paired = sorted(zip(positions, vectors), key=lambda x: x[0])

            chunks_embed = []
            chunks_attn = []
            chunks_label = []
            prev = 0
            for pos, vec in paired:
                # Tokens before this insert point
                if pos > prev:
                    chunks_embed.append(embed_b[prev:pos])
                    chunks_attn.append(attn_b[prev:pos])
                    chunks_label.append(label_b[prev:pos])
                # Insert virtual token
                chunks_embed.append(vec.unsqueeze(0))
                chunks_attn.append(torch.ones(1, device=embed_b.device, dtype=attn_b.dtype))
                chunks_label.append(torch.full((1,), -100, dtype=torch.long, device=embed_b.device))
                prev = pos

            # Remaining tokens
            if prev < L:
                chunks_embed.append(embed_b[prev:])
                chunks_attn.append(attn_b[prev:])
                chunks_label.append(label_b[prev:])

            new_embeds_list.append(torch.cat(chunks_embed, dim=0))
            new_attn_list.append(torch.cat(chunks_attn, dim=0))
            new_labels_list.append(torch.cat(chunks_label, dim=0))

        # Pad to same length
        max_new_len = max(e.size(0) for e in new_embeds_list)
        padded_embeds = torch.zeros(B, max_new_len, H, device=base_embeds.device, dtype=base_embeds.dtype)
        padded_attn = torch.zeros(B, max_new_len, device=base_embeds.device, dtype=attention_mask.dtype if attention_mask is not None else torch.long)
        padded_labels = torch.full((B, max_new_len), -100, device=base_embeds.device, dtype=torch.long)

        for b in range(B):
            n = new_embeds_list[b].size(0)
            padded_embeds[b, :n] = new_embeds_list[b]
            padded_attn[b, :n] = new_attn_list[b]
            padded_labels[b, :n] = new_labels_list[b]

        outputs = self.base_model(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attn,
            output_hidden_states=False,
        )
        return outputs.logits, padded_labels

    def forward_stage1(self, input_ids, attention_mask, labels,
                       boundary_positions, inject_positions):
        """Stage 1 forward: frozen LLM, train Proj only.

        Returns:
            loss: scalar LM loss on step tokens.
            t: (B, num_steps, H+1) hyperboloid points.
        """
        # Pass 1: collect hidden states (no grad through LLM)
        hidden_states = self.get_hidden_states(input_ids, attention_mask)

        # Compute planning vectors
        t, z, valid_mask = self.compute_plan_vectors(hidden_states, boundary_positions)

        # Pass 2: insert planning vectors as virtual tokens and compute loss
        logits, new_labels = self.insert_plan_vectors(
            input_ids, z, inject_positions, valid_mask, attention_mask, labels
        )

        # Compute LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, t
