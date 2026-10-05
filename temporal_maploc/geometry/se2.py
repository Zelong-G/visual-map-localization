"""Differentiable SE(2) operations using the repository-wide pose convention.

A pose is ``[x, y, yaw]`` and maps a point from its ego frame to a map frame.
``x`` is forward, ``y`` is left, and yaw is counter-clockwise in radians.
All functions accept tensors whose last dimension has length three; leading
dimensions follow normal PyTorch broadcasting rules.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _check_pose(pose: Tensor, name: str) -> Tensor:
    value = torch.as_tensor(pose)
    if value.shape[-1:] != (3,):
        raise ValueError(f"{name} must end in [x, y, yaw], got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        value = value.float()
    return value


def wrap_angle(angle: Tensor | float) -> Tensor | float:
    """Wrap radians to ``[-pi, pi)`` without changing tensor dtype or device."""

    if isinstance(angle, Tensor):
        return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def compose(first: Tensor, second: Tensor) -> Tensor:
    """Compose two poses: apply ``second`` in the frame defined by ``first``.

    Args:
        first: Pose tensor ending in ``[x, y, yaw]``.
        second: Pose tensor ending in ``[x, y, yaw]``.

    Returns:
        Composed pose in the map frame, with wrapped yaw in radians.
    """

    a, b = _check_pose(first, "first"), _check_pose(second, "second")
    cosine, sine = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = a[..., 0] + cosine * b[..., 0] - sine * b[..., 1]
    y = a[..., 1] + sine * b[..., 0] + cosine * b[..., 1]
    return torch.stack((x, y, torch.as_tensor(wrap_angle(a[..., 2] + b[..., 2]))), dim=-1)


def inverse(pose: Tensor) -> Tensor:
    """Return the inverse pose, mapping map-frame coordinates into ego frame."""

    value = _check_pose(pose, "pose")
    cosine, sine = torch.cos(value[..., 2]), torch.sin(value[..., 2])
    x = -cosine * value[..., 0] - sine * value[..., 1]
    y = sine * value[..., 0] - cosine * value[..., 1]
    return torch.stack((x, y, torch.as_tensor(wrap_angle(-value[..., 2]))), dim=-1)


def between(source: Tensor, target: Tensor) -> Tensor:
    """Return the pose that maps the ``source`` ego frame to ``target`` ego frame."""

    return compose(inverse(source), target)


def transform_points(pose: Tensor, points: Tensor) -> Tensor:
    """Transform planar points with an SE(2) pose.

    Args:
        pose: Pose tensor ending in ``[x, y, yaw]``.
        points: Point tensor ending in ``[x, y]`` in the pose's input frame.

    Returns:
        Points in the pose's output frame with the same final shape as
        ``points``. Distances are in meters.
    """

    value = _check_pose(pose, "pose")
    xy = torch.as_tensor(points, device=value.device, dtype=value.dtype)
    if xy.shape[-1:] != (2,):
        raise ValueError(f"points must end in [x, y], got {tuple(xy.shape)}")
    while value.ndim < xy.ndim:
        value = value.unsqueeze(-2)
    cosine, sine = torch.cos(value[..., 2]), torch.sin(value[..., 2])
    x = cosine * xy[..., 0] - sine * xy[..., 1] + value[..., 0]
    y = sine * xy[..., 0] + cosine * xy[..., 1] + value[..., 1]
    return torch.stack((x, y), dim=-1)
