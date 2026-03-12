from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM

from src.data.schema import Span
from src.model.planning_head import PlanningHead


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    lowered = dtype_name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


@dataclass
class SegmentLoss:
    loss_sum: torch.Tensor
    token_count: int


class QwenPlanningModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        torch_dtype = resolve_torch_dtype(model_cfg.get("dtype"))

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            torch_dtype=torch_dtype,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            attn_implementation=model_cfg.get("attn_implementation"),
        )
        self.backbone.config.use_cache = False

        if model_cfg.get("gradient_checkpointing", True):
            self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        model_dtype = next(self.backbone.parameters()).dtype
        self.planning_head = PlanningHead(
            hidden_size=hidden_size,
            mlp_ratio=model_cfg.get("planning_head_mlp_ratio", 4),
            dropout=model_cfg.get("planning_head_dropout", 0.0),
        ).to(dtype=model_dtype)
        self.lora_enabled = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def enable_lora(self, lora_cfg: dict[str, Any]) -> None:
        if self.lora_enabled:
            return
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            ),
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        self.lora_enabled = True

    def set_stage(self, stage: str) -> None:
        if stage == "stage1":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.planning_head.parameters():
                parameter.requires_grad = True
            return

        if stage == "stage2":
            for parameter in self.planning_head.parameters():
                parameter.requires_grad = True
            if not self.lora_enabled:
                raise RuntimeError("LoRA must be enabled before entering stage2.")
            return

        if stage == "baseline":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = parameter.requires_grad
            for parameter in self.planning_head.parameters():
                parameter.requires_grad = False
            return

        raise ValueError(f"Unsupported stage: {stage}")

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _encode_context(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.numel() == 0:
            raise ValueError("context_ids must not be empty; planning requires question (or prior) context.")

        outputs = self.backbone(
            input_ids=context_ids.unsqueeze(0),
            attention_mask=torch.ones(1, context_ids.shape[0], device=context_ids.device, dtype=torch.long),
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs.hidden_states[-1][:, -1, :]

    def _token_loss_from_embeds(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        plan_vector: torch.Tensor | None,
    ) -> SegmentLoss:
        input_embeddings = self.backbone.get_input_embeddings()
        context_embeds = (
            input_embeddings(context_ids.unsqueeze(0))
            if context_ids.numel() > 0
            else torch.empty(
                (1, 0, input_embeddings.embedding_dim),
                device=self.device,
                dtype=input_embeddings.weight.dtype,
            )
        )
        target_embeds = input_embeddings(target_ids.unsqueeze(0))

        embed_parts = [context_embeds]
        prefix_length = context_embeds.shape[1]

        if plan_vector is not None:
            embed_parts.append(plan_vector.unsqueeze(1))
            prefix_length += 1

        embed_parts.append(target_embeds)
        inputs_embeds = torch.cat(embed_parts, dim=1)

        total_length = inputs_embeds.shape[1]
        labels = torch.full((1, total_length), -100, device=self.device, dtype=torch.long)
        labels[:, prefix_length:] = target_ids.unsqueeze(0)
        attention_mask = torch.ones((1, total_length), device=self.device, dtype=torch.long)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        token_count = int(target_ids.shape[0])
        return SegmentLoss(loss_sum=outputs.loss * token_count, token_count=token_count)

    def _planning_segment_loss(self, full_ids: torch.Tensor, step_span: Span) -> SegmentLoss:
        context_ids = full_ids[: step_span.start]
        target_ids = full_ids[step_span.start : step_span.end]
        context_hidden = self._encode_context(context_ids)
        plan_vector = self.planning_head(context_hidden)
        return self._token_loss_from_embeds(context_ids, target_ids, plan_vector=plan_vector)

    def _answer_segment_loss(self, full_ids: torch.Tensor, answer_span: Span) -> SegmentLoss:
        context_ids = full_ids[: answer_span.start]
        target_ids = full_ids[answer_span.start : answer_span.end]
        return self._token_loss_from_embeds(context_ids, target_ids, plan_vector=None)

    def compute_stage1_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        total_loss_sum = torch.zeros((), device=self.device)
        total_tokens = 0

        for batch_index in range(batch["input_ids"].shape[0]):
            full_ids = batch["input_ids"][batch_index][batch["attention_mask"][batch_index].bool()].to(self.device)
            for step_span in batch["step_spans"][batch_index]:
                segment = self._planning_segment_loss(full_ids, step_span)
                total_loss_sum = total_loss_sum + segment.loss_sum
                total_tokens += segment.token_count

        if total_tokens == 0:
            raise RuntimeError("Stage1 batch has no reasoning step tokens.")

        loss = total_loss_sum / total_tokens
        return {"loss": loss, "token_count": torch.tensor(total_tokens, device=self.device)}

    def compute_stage2_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        total_loss_sum = torch.zeros((), device=self.device)
        total_tokens = 0

        for batch_index in range(batch["input_ids"].shape[0]):
            full_ids = batch["input_ids"][batch_index][batch["attention_mask"][batch_index].bool()].to(self.device)

            for step_span in batch["step_spans"][batch_index]:
                segment = self._planning_segment_loss(full_ids, step_span)
                total_loss_sum = total_loss_sum + segment.loss_sum
                total_tokens += segment.token_count

            answer_segment = self._answer_segment_loss(full_ids, batch["answer_spans"][batch_index])
            total_loss_sum = total_loss_sum + answer_segment.loss_sum
            total_tokens += answer_segment.token_count

        if total_tokens == 0:
            raise RuntimeError("Stage2 batch has no supervised tokens.")

        loss = total_loss_sum / total_tokens
        return {"loss": loss, "token_count": torch.tensor(total_tokens, device=self.device)}

    def compute_baseline_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        token_count = int((labels != -100).sum().item())
        return {
            "loss": outputs.loss,
            "token_count": torch.tensor(token_count, device=self.device),
        }

    @torch.no_grad()
    def generate_step(
        self,
        context_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_id: int | None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        generated: list[int] = []
        base_context = context_ids.to(self.device)
        plan_vector = self.planning_head(self._encode_context(base_context))

        for _ in range(max_new_tokens):
            current_target = torch.tensor(generated, device=self.device, dtype=torch.long)
            input_embeddings = self.backbone.get_input_embeddings()
            context_embeds = (
                input_embeddings(base_context.unsqueeze(0))
                if base_context.numel() > 0
                else torch.empty(
                    (1, 0, input_embeddings.embedding_dim),
                    device=self.device,
                    dtype=input_embeddings.weight.dtype,
                )
            )
            target_embeds = (
                input_embeddings(current_target.unsqueeze(0))
                if current_target.numel() > 0
                else torch.empty(
                    (1, 0, input_embeddings.embedding_dim),
                    device=self.device,
                    dtype=input_embeddings.weight.dtype,
                )
            )
            inputs_embeds = torch.cat([context_embeds, plan_vector.unsqueeze(1), target_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)
            logits = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :]

            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated.append(token_id)

            if stop_token_id is not None and token_id == stop_token_id:
                break

        return torch.tensor(generated, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def generate_answer(
        self,
        context_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        generated: list[int] = []
        current_context = context_ids.to(self.device)

        for _ in range(max_new_tokens):
            target_ids = torch.tensor(generated, device=self.device, dtype=torch.long)
            input_ids = (
                torch.cat([current_context, target_ids], dim=0).unsqueeze(0)
                if target_ids.numel() > 0
                else current_context.unsqueeze(0)
            )
            attention_mask = torch.ones_like(input_ids)
            logits = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, -1, :]

            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated.append(token_id)

            if eos_token_id is not None and token_id == eos_token_id:
                break

        return torch.tensor(generated, device=self.device, dtype=torch.long)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)
