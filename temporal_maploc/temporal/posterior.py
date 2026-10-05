"""Probability operations for discrete SE(2) pose posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from temporal_maploc.models.pose_solver import PoseGrid


def normalize_probabilities(probabilities: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    """Normalize nonnegative candidate masses along the final dimension."""

    value = torch.as_tensor(probabilities)
    if value.ndim < 1 or not torch.is_floating_point(value):
        raise ValueError("probabilities must be a floating tensor with a candidate dimension")
    if not torch.isfinite(value).all() or torch.any(value < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    mass = value.sum(dim=-1, keepdim=True)
    if torch.any(mass <= eps):
        raise ValueError("posterior has zero probability mass")
    return value / mass


def posterior_expectation(probabilities: Tensor, grid: PoseGrid) -> Tensor:
    """Return posterior mean corrections ``[B,3]`` with circular yaw mean."""

    probs = normalize_probabilities(probabilities)
    if probs.shape[-1] != grid.count:
        raise ValueError("probability size does not match pose grid")
    offsets = grid.offsets.to(probs)
    xy = probs @ offsets[:, :2]
    yaw = torch.atan2(probs @ torch.sin(offsets[:, 2]), probs @ torch.cos(offsets[:, 2]))
    return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)


def posterior_entropy(probabilities: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    """Return discrete entropy ``[B]`` in nats for normalized posteriors."""

    probs = normalize_probabilities(probabilities)
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)


@dataclass(frozen=True)
class PosePosterior:
    """Normalized discrete posterior over one :class:`PoseGrid`."""

    probabilities: Tensor
    grid: PoseGrid

    def __post_init__(self) -> None:
        probabilities = normalize_probabilities(self.probabilities)
        if probabilities.ndim != 2 or probabilities.shape[-1] != self.grid.count:
            raise ValueError("posterior probabilities must be [B,K] for the supplied grid")
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def estimate(self) -> Tensor:
        """Posterior expected correction ``[B,3]`` in meters/radians."""

        return posterior_expectation(self.probabilities, self.grid)
