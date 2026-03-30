"""Lorentz (hyperboloid) model of hyperbolic space."""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7
CLAMP_MAX = 50.0


def lorentz_origin(dim: int, device=None, dtype=None) -> torch.Tensor:
    """Origin point on the hyperboloid: o = (1, 0, ..., 0) in R^{dim+1}."""
    o = torch.zeros(dim + 1, device=device, dtype=dtype)
    o[0] = 1.0
    return o


def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product: <x, y>_L = -x_0*y_0 + sum(x_i*y_i for i>=1).

    Args:
        x, y: (..., d+1) tensors on the hyperboloid.
    Returns:
        (...,) tensor of inner products.
    """
    time = -x[..., 0] * y[..., 0]
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return time + space


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance: d_H(x, y) = arccosh(-<x, y>_L).

    Args:
        x, y: (..., d+1) tensors on the hyperboloid.
    Returns:
        (...,) tensor of distances.
    """
    inner = minkowski_inner(x, y)
    # -<x, y>_L should be >= 1 for valid hyperboloid points; clamp for stability
    return torch.acosh(torch.clamp(-inner, min=1.0 + EPS))


def exp_map_origin(v: torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin of the Lorentz model.

    Maps tangent vector v in R^d to a point on the hyperboloid in R^{d+1}.

    Args:
        v: (..., d) tangent vector at the origin.
    Returns:
        (..., d+1) point on the hyperboloid.
    """
    v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=EPS, max=CLAMP_MAX)
    # For small norms, use Taylor expansion: cosh(x) ≈ 1, sinh(x)/x ≈ 1
    small_mask = (v_norm < 1e-5)
    coeff = torch.where(small_mask, torch.ones_like(v_norm), torch.sinh(v_norm) / v_norm)
    time_component = torch.where(small_mask, torch.ones_like(v_norm), torch.cosh(v_norm))
    space_component = coeff * v  # (..., d)
    return torch.cat([time_component, space_component], dim=-1)  # (..., d+1)


def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """Project a point in R^{d+1} onto the hyperboloid (for numerical correction).

    Ensures -x_0^2 + ||x_{1:}||^2 = -1 by fixing x_0 = sqrt(1 + ||x_{1:}||^2).
    """
    space = x[..., 1:]
    space_sq_norm = (space * space).sum(dim=-1, keepdim=True)
    time_component = torch.sqrt(torch.clamp(1.0 + space_sq_norm, min=EPS))
    return torch.cat([time_component, space], dim=-1)
