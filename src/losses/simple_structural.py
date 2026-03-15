from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_simple_structural_losses(
    segment_logits: torch.Tensor,
    segment_targets: torch.Tensor,
    depth_preds: torch.Tensor,
    depth_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if segment_logits.numel() == 0:
        zero = depth_targets.new_zeros(())
        return zero, zero

    seg_loss = F.cross_entropy(segment_logits, segment_targets)
    depth_loss = F.mse_loss(depth_preds, depth_targets)
    return seg_loss, depth_loss
