from __future__ import annotations

import torch

from temporal_maploc.models.pose_solver import PoseGrid
from temporal_maploc.temporal.fusion import fuse_posteriors
from temporal_maploc.temporal.pose_propagation import propagate_pose_posterior


def test_motion_transport_moves_probability_mass() -> None:
    grid = PoseGrid.from_ranges(x_m=(-2.0, 2.0, 1.0), y_m=(0.0, 0.0, 1.0), yaw_deg=(0.0, 0.0, 1.0))
    previous = torch.zeros(1, grid.count)
    previous[0, 2] = 1.0  # x=0, y=0, yaw=0
    result = propagate_pose_posterior(
        previous,
        grid=grid,
        previous_initial_pose=torch.zeros(1, 3),
        current_initial_pose=torch.zeros(1, 3),
        motion_mean=torch.tensor([[1.0, 0.0, 0.0]]),
        motion_covariance=torch.zeros(1, 3, 3),
        covariance_floor=(0.0, 0.0, 0.0),
    )
    assert torch.allclose(result.prior.sum(dim=-1), torch.ones(1), atol=1.0e-6)
    assert torch.allclose(result.retained_mass, torch.ones(1), atol=1.0e-6)
    assert int(result.prior.argmax(dim=-1)) == 3  # x=1 on a y/yaw singleton grid


def test_out_of_grid_transport_is_finite_and_fusion_normalizes() -> None:
    grid = PoseGrid.from_ranges(x_m=(-1.0, 1.0, 1.0), y_m=(0.0, 0.0, 1.0), yaw_deg=(0.0, 0.0, 1.0))
    previous = torch.full((1, grid.count), 1.0 / grid.count)
    result = propagate_pose_posterior(
        previous,
        grid=grid,
        previous_initial_pose=torch.zeros(1, 3),
        current_initial_pose=torch.zeros(1, 3),
        motion_mean=torch.tensor([[20.0, 0.0, 0.0]]),
        motion_covariance=torch.zeros(1, 3, 3),
        covariance_floor=(0.0, 0.0, 0.0),
    )
    fused = fuse_posteriors(previous, result.prior, prior_weight=0.8)
    assert torch.isfinite(result.prior).all() and torch.isfinite(fused).all()
    assert torch.allclose(fused.sum(dim=-1), torch.ones(1), atol=1.0e-6)
