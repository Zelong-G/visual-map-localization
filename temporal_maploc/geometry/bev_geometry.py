"""Metric conversion and sampling for bird's-eye-view feature tensors.

The convention is deliberately explicit: BEV rows represent decreasing
forward ``x`` and BEV columns represent increasing leftward ``y``. This
matches a top-down image with forward at the top and left at the right.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True)
class BEVGeometry:
    """Physical footprint of a BEV tensor.

    ``x_range_m`` and ``y_range_m`` are inclusive physical bounds in meters.
    Tensor cell centers are linearly distributed across these bounds.
    """

    x_range_m: tuple[float, float] = (-40.0, 40.0)
    y_range_m: tuple[float, float] = (-40.0, 40.0)

    def __post_init__(self) -> None:
        if self.x_range_m[0] >= self.x_range_m[1] or self.y_range_m[0] >= self.y_range_m[1]:
            raise ValueError("BEV metric ranges must have positive extent")

    def metric_to_grid(self, points_xy: Tensor) -> Tensor:
        """Map local metric points ``[...,2]`` to grid-sample coordinates.

        Returns ``[...,2]`` in ``(column, row)`` order and in ``[-1,1]``;
        out-of-range values are retained so that ``grid_sample`` can apply its
        requested padding behavior.
        """

        points = torch.as_tensor(points_xy)
        if points.shape[-1:] != (2,):
            raise ValueError(f"points must end in [x, y], got {tuple(points.shape)}")
        x_min, x_max = self.x_range_m
        y_min, y_max = self.y_range_m
        col = 2.0 * (points[..., 1] - y_min) / (y_max - y_min) - 1.0
        row = 2.0 * (x_max - points[..., 0]) / (x_max - x_min) - 1.0
        return torch.stack((col, row), dim=-1)


def sample_bev(
    feature: Tensor,
    points_xy: Tensor,
    geometry: BEVGeometry,
    *,
    padding_mode: str = "zeros",
) -> Tensor:
    """Bilinearly sample BEV features at local metric points.

    Args:
        feature: BEV tensor ``[B,C,H,W]``.
        points_xy: Metric locations ``[B,N,2]`` in meters.
        geometry: Coordinate conversion for the BEV tensor.
        padding_mode: A valid PyTorch ``grid_sample`` padding mode.

    Returns:
        Sampled features of shape ``[B,N,C]``.
    """

    if feature.ndim != 4:
        raise ValueError(f"feature must be [B,C,H,W], got {tuple(feature.shape)}")
    points = torch.as_tensor(points_xy, device=feature.device, dtype=feature.dtype)
    if points.ndim != 3 or points.shape[0] != feature.shape[0] or points.shape[-1] != 2:
        raise ValueError("points must be [B,N,2] with the same batch size as feature")
    grid = geometry.metric_to_grid(points).unsqueeze(2)
    sampled = functional.grid_sample(
        feature, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )
    return sampled.squeeze(-1).transpose(1, 2)
