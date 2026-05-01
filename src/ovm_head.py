"""OVM value head.

A scalar token-level value head for outcome-supervised value modeling.
The head reads the last-layer hidden state at every token position and
predicts a value in [0,1] = "probability that continuing from this prefix
yields the correct final answer." Training is per-token MSE against the
outcome label of the rollout the prefix belongs to.

Architecture:
    h_t (hidden_dim) → linear → scalar logit → sigmoid → v_t in [0,1]

The head is a single nn.Linear with bias. We use the BASE model frozen
(plus the existing PT-SFT LoRA frozen) and train only this scalar head.
This matches the OVM paper's "verifier" setup, which initializes from
the generator and adds a small head, except we keep the body frozen
since our PT-SFT LoRA is already fit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Linear scalar head: hidden_dim → 1, then sigmoid."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, seq, hidden_dim) or (batch, hidden_dim).
        Returns: same shape minus last dim, values in (0, 1)."""
        logits = self.linear(h.float()).squeeze(-1)
        return torch.sigmoid(logits)
