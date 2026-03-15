from __future__ import annotations

import torch
import torch.nn.functional as F


def _info_nce_loss(
    vectors: torch.Tensor,
    segment_ids_raw: torch.Tensor,
    solution_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    n = vectors.size(0)
    if n <= 1:
        return vectors.new_zeros(())

    x = F.normalize(vectors, dim=-1)
    sim = x @ x.T
    sim = sim / temperature

    eye = torch.eye(n, device=vectors.device, dtype=torch.bool)
    same_segment = segment_ids_raw[:, None] == segment_ids_raw[None, :]
    same_solution = solution_ids[:, None] == solution_ids[None, :]
    positive = same_segment & same_solution & ~eye

    valid_anchor = positive.any(dim=1)
    if not valid_anchor.any():
        return vectors.new_zeros(())

    sim = sim.masked_fill(eye, -1e9)
    log_denom = torch.logsumexp(sim, dim=1)

    sim_pos = sim.masked_fill(~positive, -1e9)
    log_pos = torch.logsumexp(sim_pos, dim=1)

    losses = -(log_pos - log_denom)
    return losses[valid_anchor].mean()


def _monotonic_hinge_loss(
    depth_vectors: torch.Tensor,
    segment_ids_raw: torch.Tensor,
    depth_targets: torch.Tensor,
    solution_ids: torch.Tensor,
    depth_readout,
    margin: float = 1.0,
) -> torch.Tensor:
    if depth_vectors.numel() == 0:
        return depth_vectors.new_zeros(())

    scores = depth_readout(depth_vectors).squeeze(-1)
    losses = []
    for i in range(depth_vectors.size(0)):
        for j in range(depth_vectors.size(0)):
            if i == j:
                continue
            if solution_ids[i] != solution_ids[j]:
                continue
            if segment_ids_raw[i] != segment_ids_raw[j]:
                continue
            if depth_targets[i] >= depth_targets[j]:
                continue
            losses.append(F.relu(scores[i] - scores[j] + margin))

    if not losses:
        return depth_vectors.new_zeros(())
    return torch.stack(losses).mean()


def compute_contrastive_structural_losses(
    plan_vectors: torch.Tensor,
    segment_ids_raw: torch.Tensor,
    depth_targets: torch.Tensor,
    solution_ids: torch.Tensor,
    depth_readout,
    temperature: float = 0.1,
    margin: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if plan_vectors.numel() == 0:
        zero = depth_targets.new_zeros(())
        return zero, zero

    half = plan_vectors.size(-1) // 2
    seg_vectors = plan_vectors[:, :half]
    depth_vectors = plan_vectors[:, half:]

    seg_loss = _info_nce_loss(
        vectors=seg_vectors,
        segment_ids_raw=segment_ids_raw,
        solution_ids=solution_ids,
        temperature=temperature,
    )
    depth_loss = _monotonic_hinge_loss(
        depth_vectors=depth_vectors,
        segment_ids_raw=segment_ids_raw,
        depth_targets=depth_targets,
        solution_ids=solution_ids,
        depth_readout=depth_readout,
        margin=margin,
    )
    return seg_loss, depth_loss
