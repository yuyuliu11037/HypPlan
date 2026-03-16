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

    def _auxdec_loss(
        self,
        aux_decoder: nn.Module | None,
        batch: dict[str, Any],
        t_flat: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if aux_decoder is None or t_flat.numel() == 0:
            return torch.zeros((), device=device)

        embed_layer = self.base_model.get_input_embeddings()
        aux_dtype = next(aux_decoder.parameters()).dtype
        losses = []
        offset = 0
        for b_idx, spans in enumerate(batch["step_spans"]):
            for step_idx, (start, end) in enumerate(spans):
                tok_ids = batch["input_ids"][b_idx, start:end].unsqueeze(0)
                step_embeds = embed_layer(tok_ids).to(dtype=aux_dtype)
                target = tok_ids.clone()
                losses.append(aux_decoder(t_flat[offset].unsqueeze(0).to(dtype=aux_dtype), step_embeds, target))
                offset += 1
        if not losses:
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()

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
        lambda_aux: float = 1.0,
        lambda_seg: float = 0.1,
        lambda_depth: float = 0.1,
        aux_decoder: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        _, t_flat = self._collect_plan_vectors(last_hidden, batch["input_ids"])
        hidden_with_plan = self._apply_hidden_injection(last_hidden, batch["input_ids"], t_flat)
        logits = self.base_model.lm_head(hidden_with_plan)

        lm_loss = masked_causal_lm_loss(logits, batch["labels_stage1"])
        aux_loss = self._auxdec_loss(aux_decoder, batch, t_flat, device=lm_loss.device)
        seg_loss, depth_loss = self._structural_losses(batch, t_flat, device=lm_loss.device)
        total = lm_loss + lambda_aux * aux_loss + lambda_seg * seg_loss + lambda_depth * depth_loss
        return {
            "loss": total,
            "lm_loss": lm_loss.detach(),
            "aux_loss": aux_loss.detach(),
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

        injected = input_embeds.clone()
        for b_idx in range(batch["input_ids"].size(0)):
            plan_positions = torch.where(batch["input_ids"][b_idx] == self.plan_token_id)[0]
            for local_idx, plan_pos in enumerate(plan_positions):
                inject_pos = plan_pos + 1
                if inject_pos >= batch["input_ids"].size(1):
                    continue
                flat_index = sum(
                    torch.where(batch["input_ids"][k] == self.plan_token_id)[0].numel()
                    for k in range(b_idx)
                ) + local_idx
                if flat_index < t_flat.size(0):
                    injected[b_idx, inject_pos, :] = (
                        injected[b_idx, inject_pos, :] + t_flat[flat_index].to(dtype=injected.dtype)
                    )

        pass2 = self.base_model(
            inputs_embeds=injected,
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        lm_loss = masked_causal_lm_loss(pass2.logits, batch["labels_stage2"])
        return {"loss": lm_loss, "lm_loss": lm_loss.detach()}

