"""Projection MLP: maps LLM hidden states to Lorentz hyperbolic space."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.hyperbolic import exp_map_origin


class ProjMLP(nn.Module):
    """Maps LLM hidden states to Lorentz hyperbolic space.

    Architecture: hidden_dim -> proj_hidden_dims -> hyp_dim (tangent vector)
    Then applies exp_map_origin to get a point on the hyperboloid.
    """

    def __init__(self, hidden_dim: int, hyp_dim: int, proj_hidden_dims: list[int]):
        super().__init__()
        layers = []
        in_dim = hidden_dim
        for out_dim in proj_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, hyp_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (..., hidden_dim) LLM hidden state.
        Returns:
            t: (..., hyp_dim+1) point on the hyperboloid.
            z: (..., hyp_dim) tangent vector at origin.
        """
        z = self.mlp(h)
        t = exp_map_origin(z)
        return t, z


class ProjectBack(nn.Module):
    """Projects tangent vectors back to LLM embedding space for injection."""

    def __init__(self, hyp_dim: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(hyp_dim, embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (..., hyp_dim) tangent vector.
        Returns:
            (..., embed_dim) vector to add to token embeddings.
        """
        return self.linear(z)
