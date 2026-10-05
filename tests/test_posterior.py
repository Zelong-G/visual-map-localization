from __future__ import annotations

import torch

from temporal_maploc.models.pose_solver import PoseGrid
from temporal_maploc.temporal.posterior import PosePosterior, normalize_probabilities, posterior_entropy, posterior_expectation


def test_posterior_normalization_and_entropy_are_finite() -> None:
    grid = PoseGrid.from_ranges(x_m=(0.0, 1.0, 1.0), y_m=(0.0, 0.0, 1.0), yaw_deg=(0.0, 0.0, 1.0))
    probability = normalize_probabilities(torch.tensor([[2.0, 2.0]]))
    posterior = PosePosterior(probability, grid)
    assert torch.allclose(posterior.probabilities.sum(dim=-1), torch.ones(1))
    assert torch.allclose(posterior_expectation(probability, grid), torch.tensor([[0.5, 0.0, 0.0]]))
    assert torch.isfinite(posterior_entropy(probability)).all()
