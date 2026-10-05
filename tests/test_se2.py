from __future__ import annotations

import math

import torch

from temporal_maploc.geometry.se2 import between, compose, inverse, transform_points, wrap_angle


def test_compose_inverse_is_identity() -> None:
    pose = torch.tensor([[2.0, -1.0, 0.7], [-3.0, 4.0, -2.1]])
    identity = compose(pose, inverse(pose))
    assert torch.allclose(identity[:, :2], torch.zeros_like(identity[:, :2]), atol=1.0e-6)
    assert torch.allclose(identity[:, 2], torch.zeros_like(identity[:, 2]), atol=1.0e-6)


def test_between_recovers_relative_pose() -> None:
    source = torch.tensor([[1.0, 2.0, math.pi / 2]])
    target = torch.tensor([[1.0, 3.0, math.pi / 2]])
    relative = between(source, target)
    assert torch.allclose(relative, torch.tensor([[1.0, 0.0, 0.0]]), atol=1.0e-6)


def test_transform_points_and_yaw_wrap() -> None:
    pose = torch.tensor([[1.0, 0.0, math.pi / 2]])
    points = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    transformed = transform_points(pose, points)
    assert torch.allclose(transformed, torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]), atol=1.0e-6)
    assert float(wrap_angle(3.0 * math.pi)) == -math.pi
