from __future__ import annotations

import torch
import torch.nn.functional as F


def simple_structural_losses(
    t_seg: torch.Tensor,
    t_depth: torch.Tensor,
    segment_logits: torch.Tensor,
    depth_preds: torch.Tensor,
    segment_targets: torch.Tensor,
    depth_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seg_loss = F.cross_entropy(segment_logits, segment_targets)
    depth_loss = F.mse_loss(depth_preds.squeeze(-1), depth_targets.float())
    return seg_loss, depth_loss

