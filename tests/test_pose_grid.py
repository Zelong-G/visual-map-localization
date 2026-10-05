from __future__ import annotations

import math

import torch

from temporal_maploc.models.pose_solver import PoseGrid


def test_grid_has_inclusive_axes_and_cartesian_offsets() -> None:
    grid = PoseGrid.from_ranges(x_m=(-1.0, 1.0, 1.0), y_m=(0.0, 1.0, 1.0), yaw_deg=(-10.0, 10.0, 10.0))
    assert grid.shape == (3, 2, 3)
    assert grid.count == 18
    assert torch.allclose(grid.offsets[0], torch.tensor([-1.0, 0.0, math.radians(-10.0)]))
    assert torch.allclose(grid.offsets[-1], torch.tensor([1.0, 1.0, math.radians(10.0)]))


def test_flatten_round_trip() -> None:
    grid = PoseGrid.from_ranges(x_m=(0.0, 1.0, 1.0), y_m=(0.0, 1.0, 1.0), yaw_deg=(0.0, 0.0, 1.0))
    volume = torch.arange(grid.count, dtype=torch.float32).reshape(grid.shape)
    assert torch.equal(grid.unflatten(grid.flatten(volume)), volume)
