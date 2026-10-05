"""Sensor-agnostic planar motion representation and twist integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from temporal_maploc.geometry.se2 import wrap_angle


@dataclass(frozen=True)
class RelativeMotion:
    """A causal relative SE(2) motion estimate with uncertainty.

    ``mean`` is ``[B,3] = [dx, dy, dyaw]`` from the previous ego frame to the
    current ego frame. Distances are meters, yaw is radians, and covariance is
    ``[B,3,3]`` in the same units squared. ``reliability`` is in ``[0,1]``.
    """

    mean: Tensor
    covariance: Tensor
    reliability: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean)
        covariance = torch.as_tensor(self.covariance, device=mean.device, dtype=mean.dtype)
        reliability = torch.as_tensor(self.reliability, device=mean.device, dtype=mean.dtype)
        if mean.ndim != 2 or mean.shape[-1] != 3:
            raise ValueError("motion mean must be [B,3]")
        if covariance.shape != (mean.shape[0], 3, 3):
            raise ValueError("motion covariance must be [B,3,3]")
        if reliability.shape not in ((mean.shape[0],), (mean.shape[0], 1)):
            raise ValueError("motion reliability must be [B] or [B,1]")
        if not torch.is_floating_point(mean) or not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise ValueError("motion tensors must be finite floating point")
        if torch.any(torch.diagonal(covariance, dim1=-2, dim2=-1) < 0.0):
            raise ValueError("motion covariance diagonal must be nonnegative")
        if torch.any((reliability < 0.0) | (reliability > 1.0)):
            raise ValueError("motion reliability must lie in [0,1]")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", 0.5 * (covariance + covariance.transpose(-1, -2)))
        object.__setattr__(self, "reliability", reliability.reshape(-1))


def integrate_planar_twist(
    speed_m_s: Tensor,
    yaw_rate_rad_s: Tensor,
    duration_s: Tensor,
    *,
    std: tuple[float, float, float] = (0.10, 0.10, 0.02),
) -> RelativeMotion:
    """Integrate a constant planar vehicle twist into a relative motion prior.

    Args:
        speed_m_s: Forward speed ``[B]`` in meters per second.
        yaw_rate_rad_s: Counter-clockwise yaw rate ``[B]`` in radians/s.
        duration_s: Positive integration interval ``[B]`` in seconds.
        std: Conservative standard deviations for `(x, y, yaw)`.

    Returns:
        A :class:`RelativeMotion` in the previous ego frame. This utility has
        no access to ground truth, future frames, or dataset-specific CAN APIs.
    """

    speed = torch.as_tensor(speed_m_s)
    rate = torch.as_tensor(yaw_rate_rad_s, dtype=speed.dtype, device=speed.device)
    duration = torch.as_tensor(duration_s, dtype=speed.dtype, device=speed.device)
    if speed.ndim != 1 or rate.shape != speed.shape or duration.shape != speed.shape:
        raise ValueError("speed, yaw rate, and duration must all be [B]")
    if not torch.isfinite(speed).all() or not torch.isfinite(rate).all() or torch.any(duration <= 0.0):
        raise ValueError("twist inputs must be finite and duration positive")
    yaw = rate * duration
    small = rate.abs() < 1.0e-6
    radius = speed / rate.clamp_min(1.0e-6).where(rate >= 0.0, rate.clamp_max(-1.0e-6))
    dx_arc = radius * torch.sin(yaw)
    dy_arc = radius * (1.0 - torch.cos(yaw))
    dx = torch.where(small, speed * duration, dx_arc)
    dy = torch.where(small, torch.zeros_like(dx), dy_arc)
    mean = torch.stack((dx, dy, torch.as_tensor(wrap_angle(yaw))), dim=-1)
    scale = torch.as_tensor(std, dtype=speed.dtype, device=speed.device)
    if scale.shape != (3,) or torch.any(scale <= 0.0):
        raise ValueError("std must contain three positive values")
    covariance = torch.diag_embed(scale.square().expand(speed.shape[0], -1))
    reliability = torch.ones_like(speed)
    return RelativeMotion(mean, covariance, reliability)
