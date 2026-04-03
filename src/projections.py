"""Projection MLP: maps LLM hidden states to Lorentz hyperbolic space."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.hyperbolic import exp_map_origin


class ProjMLP(nn.Module):
    """Maps LLM hidden states to Lorentz hyperbolic space.

    Architecture: hidden_dim -> proj_hidden_dims -> hidden_dim (tangent vector z)
    Then applies exp_map_origin to get a point on the hyperboloid in R^{hidden_dim+1}.

    z has the same dimension as the LLM hidden states, so it can be directly
    inserted into the embedding sequence as a virtual planning token.
    """

    def __init__(self, hidden_dim: int, proj_hidden_dims: list[int]):
        super().__init__()
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
            z: (..., hidden_dim) tangent vector at origin (same dim as embeddings).
        """
        z = self.mlp(h)
        t = exp_map_origin(z)
        return t, z
