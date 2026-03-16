from __future__ import annotations

import torch
import torch.nn.functional as F


def segment_infonce_loss(
    embeddings: torch.Tensor,
    solution_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    x = F.normalize(embeddings, dim=-1)
    sim = x @ x.transpose(0, 1) / temperature
    sim.fill_diagonal_(-1e9)

    total = embeddings.new_zeros(())
    count = 0
    for i in range(embeddings.size(0)):
        positive_mask = (solution_ids == solution_ids[i]) & (segment_ids == segment_ids[i])
        positive_mask[i] = False
        if not torch.any(positive_mask):
            continue
        numerator = torch.logsumexp(sim[i][positive_mask], dim=0)
        denominator = torch.logsumexp(sim[i], dim=0)
        total = total - (numerator - denominator)
        count += 1
    if count == 0:
        return embeddings.new_zeros(())
    return total / count


def monotonic_hinge_loss(
    depth_scores: torch.Tensor,
    solution_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    depth_targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    losses = []
    n = depth_scores.size(0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if solution_ids[i] != solution_ids[j] or segment_ids[i] != segment_ids[j]:
                continue
            if depth_targets[i] >= depth_targets[j]:
                continue
            losses.append(F.relu(depth_scores[i] - depth_scores[j] + margin))
    if not losses:
        return depth_scores.new_zeros(())
    return torch.stack(losses).mean()

