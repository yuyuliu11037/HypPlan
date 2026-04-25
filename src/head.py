"""Hyperbolic head: LLM hidden -> point in hyperbolic space.

Supports both the Poincaré ball and the Lorentz hyperboloid as an exp-map-at-
origin. Keeps the head's output in a LOW-dim hyperbolic space (default
hyp_dim=32), matching Nickel-Kiela 2017. A separate UpProjector (trained in
stage 2, not here) lifts the low-dim point back to hidden_dim for virtual-
token injection.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.hyperbolic import EPS, CLAMP_MAX, exp_map_origin, lorentz_distance


# --- Poincaré ball ops ------------------------------------------------------

POINCARE_MAX_NORM = 1.0 - 1e-5


def poincare_exp0(v: torch.Tensor) -> torch.Tensor:
    """Exp-map at origin of the Poincaré ball: x = tanh(|v|) * v / |v|.

    v: (..., d) tangent. Returns (..., d) point with |x| < 1.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS, max=CLAMP_MAX)
    coeff = torch.tanh(v_norm) / v_norm
    x = coeff * v
    # Numerical safety: cap |x| below 1
    x_norm = x.norm(dim=-1, keepdim=True)
    scale = torch.where(x_norm >= POINCARE_MAX_NORM,
                        POINCARE_MAX_NORM / x_norm.clamp(min=EPS),
                        torch.ones_like(x_norm))
    return x * scale


def poincare_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance on the Poincaré ball (curvature -1).

    d(x, y) = arccosh(1 + 2 * |x - y|^2 / ((1 - |x|^2) * (1 - |y|^2)))
    """
    diff_sq = ((x - y) ** 2).sum(dim=-1)
    x_sq = (x ** 2).sum(dim=-1).clamp(max=1.0 - EPS)
    y_sq = (y ** 2).sum(dim=-1).clamp(max=1.0 - EPS)
    denom = (1.0 - x_sq) * (1.0 - y_sq)
    arg = 1.0 + 2.0 * diff_sq / denom.clamp(min=EPS)
    return torch.acosh(arg.clamp(min=1.0 + EPS))


# --- Head module -----------------------------------------------------------

class HyperbolicHead(nn.Module):
    """MLP feature -> tangent vector -> manifold point via exp-map at origin.

    Args:
        in_dim: input hidden dim (LLM hidden state dim, e.g. 4096).
        hyp_dim: hyperbolic embedding dim (e.g. 32).
        hidden_dims: MLP widths between in_dim and hyp_dim.
        manifold: "poincare" or "lorentz".
        init_scale: scale factor on the final linear's init to keep initial
            tangent vectors small (so exp-maps start near the origin).
    """

    def __init__(
        self,
        in_dim: int,
        hyp_dim: int = 32,
        hidden_dims: list[int] | None = None,
        manifold: Literal["poincare", "lorentz", "euclidean"] = "poincare",
        init_scale: float = 1e-3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [1024, 256]
        self.manifold = manifold
        self.hyp_dim = hyp_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        final = nn.Linear(prev, hyp_dim)
        nn.init.normal_(final.weight, std=init_scale)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., in_dim) -> point on the manifold.

        Poincaré:  returns (..., hyp_dim).
        Lorentz:   returns (..., hyp_dim + 1).
        Euclidean: returns (..., hyp_dim) — the raw MLP output; no exp-map.
        """
        v = self.mlp(h)
        if self.manifold == "poincare":
            return poincare_exp0(v)
        elif self.manifold == "lorentz":
            return exp_map_origin(v)
        elif self.manifold == "euclidean":
            return v
        else:
            raise ValueError(f"unknown manifold {self.manifold}")

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distance between two points in the chosen manifold.

        Inputs must come from this head's forward() (so they live in the
        same manifold). Euclidean uses L2.
        """
        if self.manifold == "poincare":
            return poincare_distance(x, y)
        elif self.manifold == "lorentz":
            return lorentz_distance(x, y)
        elif self.manifold == "euclidean":
            return (x - y).norm(dim=-1)
        else:
            raise ValueError(f"unknown manifold {self.manifold}")

    def origin_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Distance from origin for each point in z."""
        if self.manifold == "poincare":
            zero = torch.zeros_like(z)
            return poincare_distance(z, zero)
        elif self.manifold == "lorentz":
            zero = torch.zeros_like(z)
            zero[..., 0] = 1.0
            return lorentz_distance(z, zero)
        elif self.manifold == "euclidean":
            return z.norm(dim=-1)
        else:
            raise ValueError(f"unknown manifold {self.manifold}")


# --- UpProjector -----------------------------------------------------------

class UpProjector(nn.Module):
    """Map low-dim manifold point back to hidden_dim for virtual-token injection.

    Used only in stage 2. Treats the manifold point as a Euclidean feature
    vector (no inverse exp-map); the stage-2 LoRA learns what to do with it.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 zero_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )
        if zero_init:
            ln = self.net[-1]
            nn.init.zeros_(ln.weight)
            nn.init.zeros_(ln.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
