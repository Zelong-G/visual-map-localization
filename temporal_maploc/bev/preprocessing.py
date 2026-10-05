"""Validation and normalization for BEV feature tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from temporal_maploc.geometry.bev_geometry import BEVGeometry


@dataclass(frozen=True)
class BEVSpec:
    """Required layout and metric geometry of a BEV feature tensor.

    Features use ``[B,C,H,W]`` order. Rows decrease in forward meters and
    columns increase in leftward meters; see :class:`BEVGeometry`.
    """

    channels: int
    geometry: BEVGeometry = BEVGeometry()

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError("BEV channel count must be positive")


def validate_bev(feature: Tensor, spec: BEVSpec) -> Tensor:
    """Validate and return a finite floating ``[B,C,H,W]`` BEV tensor."""

    value = torch.as_tensor(feature)
    if value.ndim != 4 or value.shape[1] != spec.channels:
        raise ValueError(
            f"BEV must be [B,{spec.channels},H,W], got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError("BEV features must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError("BEV features contain NaN or Inf")
    return value


def normalize_bev(feature: Tensor, *, eps: float = 1.0e-6) -> Tensor:
    """Normalize each feature channel over its spatial dimensions.

    Args:
        feature: BEV tensor ``[B,C,H,W]``.
        eps: Positive standard-deviation floor.

    Returns:
        Normalized BEV tensor with unchanged shape and coordinate frame.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    value = torch.as_tensor(feature)
    mean = value.mean(dim=(-2, -1), keepdim=True)
    std = value.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (value - mean) / std
