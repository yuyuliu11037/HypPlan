from __future__ import annotations

import torch
from torch import nn


class PlanningHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        intermediate_size = hidden_size * mlp_ratio
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_state)
