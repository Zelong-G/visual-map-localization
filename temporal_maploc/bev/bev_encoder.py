"""Small trainable adapter for precomputed or externally produced BEV features."""

from __future__ import annotations

from torch import Tensor, nn

from .preprocessing import BEVSpec, normalize_bev, validate_bev


class BEVEncoder(nn.Module):
    """Project an input BEV tensor into localization feature channels.

    This module is deliberately not a camera–LiDAR perception implementation.
    It consumes a BEV tensor from a separately selected backbone and returns
    ``[B, hidden_dim, H, W]`` in the same metric coordinate system.
    """

    def __init__(self, input_spec: BEVSpec, hidden_dim: int, *, normalize: bool = True) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.input_spec = input_spec
        self.normalize = bool(normalize)
        self.projection = nn.Sequential(
            nn.Conv2d(input_spec.channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        """Encode a finite BEV tensor ``[B,C,H,W]`` into ``[B,D,H,W]``."""

        value = validate_bev(feature, self.input_spec)
        return self.projection(normalize_bev(value) if self.normalize else value)
