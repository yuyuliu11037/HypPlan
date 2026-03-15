from __future__ import annotations

import torch.nn as nn


class LinearProj(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.proj(x)


class MLPProj(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


def build_proj(proj_type: str, hidden_size: int) -> nn.Module:
    if proj_type == "linear":
        return LinearProj(hidden_size)
    if proj_type == "mlp":
        return MLPProj(hidden_size)
    raise ValueError(f"Unknown proj_type={proj_type!r}. Use 'linear' or 'mlp'.")
