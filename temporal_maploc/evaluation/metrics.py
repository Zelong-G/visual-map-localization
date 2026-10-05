"""Metric computation for SE(2) pose-correction estimates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from temporal_maploc.geometry.se2 import wrap_angle


@dataclass(frozen=True)
class PoseMetrics:
    """Mean absolute translation/yaw errors in meters and degrees."""

    dx_mae_m: float
    dy_mae_m: float
    dxy_mae_m: float
    dyaw_mae_deg: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def pose_metrics(prediction: Tensor, target: Tensor) -> PoseMetrics:
    """Compute aggregate error for ``[B,3]`` offsets in meters/radians."""

    pred = torch.as_tensor(prediction)
    truth = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[-1] != 3:
        raise ValueError("prediction and target must both be [B,3]")
    error = pred - truth
    yaw = torch.as_tensor(wrap_angle(error[:, 2])).abs().mean() * (180.0 / math.pi)
    return PoseMetrics(
        dx_mae_m=float(error[:, 0].abs().mean().detach().cpu()),
        dy_mae_m=float(error[:, 1].abs().mean().detach().cpu()),
        dxy_mae_m=float(torch.linalg.vector_norm(error[:, :2], dim=-1).mean().detach().cpu()),
        dyaw_mae_deg=float(yaw.detach().cpu()),
    )
