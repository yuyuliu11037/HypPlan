from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.contrastive_structural import monotonic_hinge_loss, segment_infonce_loss
from src.losses.simple_structural import simple_structural_losses


def masked_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def apply_plan_token_logit_delta(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    plan_token_id: int,
    plan_token_delta: torch.Tensor | None,
) -> torch.Tensor:
    if plan_token_delta is None:
        return logits
    adjusted = logits.clone()
    delta = torch.matmul(
        hidden_states.to(dtype=plan_token_delta.dtype),
        plan_token_delta,
    ).to(dtype=adjusted.dtype)
    adjusted[..., plan_token_id] = adjusted[..., plan_token_id] + delta
    return adjusted


class PlanningQwen(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        proj: nn.Module,
        plan_token_id: int,
        structural_loss: str = "simple",
        max_segments: int = 16,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.proj = proj
        self.plan_token_id = plan_token_id
        self.structural_loss = structural_loss
        hidden_size = base_model.config.hidden_size
        self.segment_head = nn.Linear(hidden_size // 2, max_segments)
        self.depth_head = nn.Linear(hidden_size // 2, 1)

    def _plan_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids == self.plan_token_id

    def _collect_plan_vectors(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._plan_positions(input_ids)
        h_plan = last_hidden[mask]
        if h_plan.numel() == 0:
            return h_plan, h_plan
        proj_dtype = next(self.proj.parameters()).dtype
        t = self.proj(h_plan.to(dtype=proj_dtype))
        return h_plan, t

    def _flat_plan_indices(self, input_ids: torch.Tensor) -> list[tuple[int, int, int]]:
        indices: list[tuple[int, int, int]] = []
        flat_index = 0
        for b_idx in range(input_ids.size(0)):
            plan_positions = torch.where(input_ids[b_idx] == self.plan_token_id)[0]
            for plan_pos in plan_positions.tolist():
                indices.append((b_idx, plan_pos, flat_index))
                flat_index += 1
        return indices

    def _apply_hidden_injection(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        t_flat: torch.Tensor,
    ) -> torch.Tensor:
        out = hidden.clone()
        mask = self._plan_positions(input_ids)
        if t_flat.numel() > 0:
            out[mask] = out[mask] + t_flat.to(dtype=out.dtype)
        return out

    def _structural_losses(self, batch: dict[str, Any], t_flat: torch.Tensor, device: torch.device):
        if t_flat.numel() == 0:
            z = torch.zeros((), device=device)
            return z, z
        hidden_half = t_flat.size(-1) // 2
        t_seg = t_flat[:, :hidden_half]
        t_depth = t_flat[:, hidden_half:]

        segment_targets = []
        depth_targets = []
        solution_ids = []
        for s_idx, (seg_list, depth_list) in enumerate(
            zip(batch["segment_ids"], batch["within_segment_depths"], strict=True)
        ):
            segment_targets.extend(seg_list)
            depth_targets.extend(depth_list)
            solution_ids.extend([s_idx] * len(seg_list))

        segment_targets_t = torch.tensor(segment_targets, device=device, dtype=torch.long)
        depth_targets_t = torch.tensor(depth_targets, device=device, dtype=t_depth.dtype)
        solution_ids_t = torch.tensor(solution_ids, device=device, dtype=torch.long)

        if self.structural_loss == "simple":
            seg_logits = self.segment_head(t_seg)
            depth_preds = self.depth_head(t_depth)
            return simple_structural_losses(
                t_seg=t_seg,
                t_depth=t_depth,
                segment_logits=seg_logits,
                depth_preds=depth_preds,
                segment_targets=segment_targets_t.clamp(min=0, max=self.segment_head.out_features - 1),
                depth_targets=depth_targets_t,
            )

        depth_scores = self.depth_head(t_depth).squeeze(-1)
        seg_loss = segment_infonce_loss(
            embeddings=t_seg,
            solution_ids=solution_ids_t,
            segment_ids=segment_targets_t,
        )
        depth_loss = monotonic_hinge_loss(
            depth_scores=depth_scores,
            solution_ids=solution_ids_t,
            segment_ids=segment_targets_t,
            depth_targets=depth_targets_t,
        )
        return seg_loss, depth_loss

    def stage1_forward(
        self,
        batch: dict[str, Any],
        reconstruct_mode: str = "contextual",
        lambda_seg: float = 0.1,
        lambda_depth: float = 0.1,
        max_step_len: int = 256,
    ) -> dict[str, torch.Tensor]:
        device = batch["input_ids"].device

        # Pass 1: compute planning vectors t_i from frozen model hidden states (no grad through base model).
        with torch.no_grad():
            pass1 = self.base_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = pass1.hidden_states[-1]

        plan_mask = self._plan_positions(batch["input_ids"])
        h_plan = last_hidden[plan_mask]
        if h_plan.numel() == 0:
            # No plan tokens present; fallback to zero reconstruction/structural losses.
            z = torch.zeros((), device=device)
            return {"loss": z, "reconstruct_loss": z, "seg_loss": z, "depth_loss": z}

        proj_dtype = next(self.proj.parameters()).dtype
        t_flat = self.proj(h_plan.to(dtype=proj_dtype))

        # Structural losses depend only on t_i.
        seg_loss, depth_loss = self._structural_losses(batch, t_flat, device=device)

        embed_layer = self.base_model.get_input_embeddings()

        if reconstruct_mode == "isolated":
            # Isolated mode: per-step reconstruction conditioned only on [PLAN] + t_i.
            # Flatten all (t_i, step_i_tokens) pairs across the batch.
            step_token_seqs: list[torch.Tensor] = []
            flat_index = 0
            for b_idx, spans in enumerate(batch["step_spans"]):
                for start, end in spans:
                    tok_ids = batch["input_ids"][b_idx, start:end]
                    if max_step_len > 0:
                        tok_ids = tok_ids[:max_step_len]
                    if tok_ids.numel() == 0:
                        continue
                    step_token_seqs.append(tok_ids)
                    flat_index += 1

            if not step_token_seqs:
                z = torch.zeros((), device=device)
                total = lambda_seg * seg_loss + lambda_depth * depth_loss
                return {"loss": total, "reconstruct_loss": z, "seg_loss": seg_loss, "depth_loss": depth_loss}

            # Pad steps to common length.
            pad_id = 0
            lengths = [seq.size(0) for seq in step_token_seqs]
            max_len = min(max(lengths), max_step_len)
            padded_steps = []
            for seq in step_token_seqs:
                seq = seq[:max_len]
                if seq.size(0) < max_len:
                    seq = torch.cat(
                        [seq, torch.full((max_len - seq.size(0),), pad_id, device=device, dtype=seq.dtype)],
                        dim=0,
                    )
                padded_steps.append(seq)
            step_ids = torch.stack(padded_steps, dim=0)  # (num_steps, max_len)

            num_steps = step_ids.size(0)
            # Compute embeddings for [PLAN] token and add t_i.
            plan_token_ids = torch.full((num_steps, 1), self.plan_token_id, device=device, dtype=torch.long)
            plan_embeds = embed_layer(plan_token_ids)
            plan_embeds = plan_embeds + t_flat[:num_steps].unsqueeze(1).to(dtype=plan_embeds.dtype)

            step_embeds = embed_layer(step_ids)
            # Inputs: [PLAN]+t_i followed by full step token embeddings.
            decoder_input_embeds = torch.cat([plan_embeds, step_embeds], dim=1)

            # Build attention mask and labels for teacher-forcing LM loss.
            attn = torch.ones((num_steps, max_len + 1), device=device, dtype=batch["attention_mask"].dtype)
            labels_full = torch.full(
                (num_steps, max_len + 1),
                -100,
                device=device,
                dtype=torch.long,
            )
            for i, length in enumerate(lengths):
                length = min(length, max_len)
                attn[i, 1 + length :] = 0
                labels_full[i, 1 : 1 + length] = step_ids[i, :length]

            pass2 = self.base_model(
                inputs_embeds=decoder_input_embeds,
                attention_mask=attn,
                use_cache=False,
                return_dict=True,
            )
            logits = pass2.logits
            reconstruct_loss = masked_causal_lm_loss(logits, labels_full)

        else:
            # Contextual mode: full-sequence reconstruction with t_i injected at [PLAN] positions.
            input_embeds = embed_layer(batch["input_ids"])
            injected = input_embeds.clone()
            flat_index = 0
            for b_idx, plan_pos, flat_index in self._flat_plan_indices(batch["input_ids"]):
                if flat_index < t_flat.size(0):
                    injected[b_idx, plan_pos, :] = (
                        injected[b_idx, plan_pos, :] + t_flat[flat_index].to(dtype=injected.dtype)
                    )

            pass2 = self.base_model(
                inputs_embeds=injected,
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            logits = pass2.logits
            reconstruct_loss = masked_causal_lm_loss(logits, batch["labels_stage1"])

        total = reconstruct_loss + lambda_seg * seg_loss + lambda_depth * depth_loss
        return {
            "loss": total,
            "reconstruct_loss": reconstruct_loss.detach(),
            "seg_loss": seg_loss.detach(),
            "depth_loss": depth_loss.detach(),
        }

    def stage2_forward(
        self,
        batch: dict[str, Any],
        plan_token_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embed_layer = self.base_model.get_input_embeddings()
        input_embeds = embed_layer(batch["input_ids"])
        if plan_token_delta is not None:
            plan_mask = self._plan_positions(batch["input_ids"]).unsqueeze(-1)
            input_embeds = input_embeds + plan_mask * plan_token_delta.view(1, 1, -1)

        with torch.no_grad():
            pass1 = self.base_model(
                inputs_embeds=input_embeds,
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            _, t_flat = self._collect_plan_vectors(pass1.hidden_states[-1], batch["input_ids"])

        # Plan CE loss: train plan_token_delta on pass1 (injection-free) hidden states
        # to match the distribution the model sees at autonomous inference time.
        # pass1 is in no_grad so hidden states are detached constants, but gradients
        # still flow to plan_token_delta through the h @ plan_token_delta dot product.
        plan_only_labels = torch.where(
            batch["labels_stage2"] == self.plan_token_id,
            batch["labels_stage2"],
            torch.full_like(batch["labels_stage2"], -100),
        )
        pass1_logits_adjusted = apply_plan_token_logit_delta(
            logits=pass1.logits,
            hidden_states=pass1.hidden_states[-1],
            plan_token_id=self.plan_token_id,
            plan_token_delta=plan_token_delta,
        )
        plan_ce_loss = masked_causal_lm_loss(pass1_logits_adjusted, plan_only_labels)

        injected = input_embeds.clone()
        for b_idx, plan_pos, flat_index in self._flat_plan_indices(batch["input_ids"]):
            if flat_index < t_flat.size(0):
                injected[b_idx, plan_pos, :] = (
                    injected[b_idx, plan_pos, :] + t_flat[flat_index].to(dtype=injected.dtype)
                )

        pass2 = self.base_model(
            inputs_embeds=injected,
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = apply_plan_token_logit_delta(
            logits=pass2.logits,
            hidden_states=pass2.hidden_states[-1],
            plan_token_id=self.plan_token_id,
            plan_token_delta=plan_token_delta,
        )
        lm_loss = masked_causal_lm_loss(logits, batch["labels_stage2"])
        total_loss = lm_loss + plan_ce_loss
        return {"loss": total_loss, "lm_loss": lm_loss.detach(), "plan_ce_loss": plan_ce_loss.detach()}

