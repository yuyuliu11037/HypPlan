from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.model.proj import build_proj


@dataclass
class PlanningForwardOutput:
    logits: torch.Tensor
    lm_loss: torch.Tensor
    plan_vectors: torch.Tensor
    plan_mask: torch.Tensor


class PlanningQwen(nn.Module):
    def __init__(
        self,
        model_name: str,
        proj_type: str = "mlp",
        structural_loss: str = "simple",
        max_segments: int = 16,
    ) -> None:
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.base_model.requires_grad_(False)

        hidden_size = self.base_model.config.hidden_size
        self.hidden_size = hidden_size
        self.structural_loss = structural_loss

        self.proj = build_proj(proj_type=proj_type, hidden_size=hidden_size)

        if structural_loss == "simple":
            self.segment_classifier = nn.Linear(hidden_size, max_segments)
            self.depth_regressor = nn.Linear(hidden_size, 1)
            self.depth_readout = None
        elif structural_loss == "contrastive":
            self.segment_classifier = None
            self.depth_regressor = None
            self.depth_readout = nn.Linear(hidden_size // 2, 1)
        else:
            raise ValueError(
                f"Unknown structural_loss={structural_loss!r}. "
                "Use 'simple' or 'contrastive'."
            )

    def resize_token_embeddings(self, new_size: int) -> None:
        self.base_model.resize_token_embeddings(new_size)

    def trainable_parameters(self):
        params = list(self.proj.parameters())
        if self.segment_classifier is not None:
            params += list(self.segment_classifier.parameters())
        if self.depth_regressor is not None:
            params += list(self.depth_regressor.parameters())
        if self.depth_readout is not None:
            params += list(self.depth_readout.parameters())
        return params

    def get_trainable_state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "proj": self.proj.state_dict(),
            "structural_loss": self.structural_loss,
        }
        if self.segment_classifier is not None:
            payload["segment_classifier"] = self.segment_classifier.state_dict()
        if self.depth_regressor is not None:
            payload["depth_regressor"] = self.depth_regressor.state_dict()
        if self.depth_readout is not None:
            payload["depth_readout"] = self.depth_readout.state_dict()
        return payload

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        self.proj.load_state_dict(state["proj"])
        if self.segment_classifier is not None and "segment_classifier" in state:
            self.segment_classifier.load_state_dict(state["segment_classifier"])
        if self.depth_regressor is not None and "depth_regressor" in state:
            self.depth_regressor.load_state_dict(state["depth_regressor"])
        if self.depth_readout is not None and "depth_readout" in state:
            self.depth_readout.load_state_dict(state["depth_readout"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_loss_mask: torch.Tensor,
        plan_positions: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> PlanningForwardOutput:
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = base_out.hidden_states[-1]

        modified_hidden = last_hidden.clone()
        plan_vectors = torch.zeros(
            (*plan_positions.shape, self.hidden_size),
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )

        batch_idx, plan_slot_idx = torch.where(plan_mask)
        if batch_idx.numel() > 0:
            token_idx = plan_positions[batch_idx, plan_slot_idx]
            h_plan = last_hidden[batch_idx, token_idx, :]
            t_plan = self.proj(h_plan)
            modified_hidden[batch_idx, token_idx, :] = h_plan + t_plan
            plan_vectors[batch_idx, plan_slot_idx, :] = t_plan

        logits = self.base_model.lm_head(modified_hidden)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = (token_loss_mask[:, 1:] & attention_mask[:, 1:].bool()).contiguous()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)

        denom = shift_mask.sum().clamp_min(1)
        lm_loss = (token_losses * shift_mask.float()).sum() / denom

        return PlanningForwardOutput(
            logits=logits,
            lm_loss=lm_loss,
            plan_vectors=plan_vectors,
            plan_mask=plan_mask,
        )
