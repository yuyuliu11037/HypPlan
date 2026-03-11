from __future__ import annotations

import torch


def normalize_weighted_loss(loss_sum: torch.Tensor, token_count: int) -> torch.Tensor:
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    return loss_sum / token_count
