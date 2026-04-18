"""Projection MLP: maps LLM hidden states to Lorentz hyperbolic space."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.hyperbolic import EPS, exp_map_origin


class ProjMLP(nn.Module):
    """Maps LLM hidden states to Lorentz hyperbolic space.

    Architecture: hidden_dim -> proj_hidden_dims -> hidden_dim (tangent vector z)
    Then applies exp_map_origin to get a point on the hyperboloid in R^{hidden_dim+1}.

    z has the same dimension as the LLM hidden states, so it can be directly
    inserted into the embedding sequence as a virtual planning token.

    If target_norm is None, skip the rescale and use the raw MLP output directly.
    """

    def __init__(self, hidden_dim: int, proj_hidden_dims: list[int],
                 target_norm: Optional[float] = 1.0):
        super().__init__()
        self.target_norm = target_norm
        layers = []
        in_dim = hidden_dim
        for out_dim in proj_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (..., hidden_dim) LLM hidden state.
        Returns:
            t: (..., hidden_dim+1) point on the hyperboloid.
            z: (..., hidden_dim) tangent vector, optionally rescaled to target_norm.
        """
        z_raw = self.mlp(h)
        t = exp_map_origin(z_raw)
        if self.target_norm is None:
            z = z_raw
        else:
            z_norm = z_raw.norm(dim=-1, keepdim=True).clamp(min=EPS)
            z = z_raw * (self.target_norm / z_norm)
        return t, z
