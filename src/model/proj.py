from __future__ import annotations

import torch.nn as nn


class ProjectionModule(nn.Module):
    def __init__(self, hidden_size: int, proj_type: str = "mlp") -> None:
        super().__init__()
        if proj_type == "linear":
            self.net = nn.Linear(hidden_size, hidden_size)
        elif proj_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

    def forward(self, hidden_state):
        return self.net(hidden_state)

