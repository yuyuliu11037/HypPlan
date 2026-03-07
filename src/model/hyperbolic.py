from __future__ import annotations

import torch
from torch import nn


class Projection(nn.Module):
    """
    Projection interface for planning latents.

    The current near-real implementation keeps projection as identity.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class IdentityProjection(Projection):
    """Alias class to make config-driven swaps explicit."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z
