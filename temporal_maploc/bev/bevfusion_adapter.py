"""Adapter boundary for an externally installed BEVFusion-style backend.

No BEVFusion source, configuration, or weight is distributed here. The caller
passes an already constructed callable that receives a batch dictionary and
returns either a BEV tensor or a mapping containing ``"bev"``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor, nn

from .preprocessing import BEVSpec, validate_bev


class BEVFusionAdapter(nn.Module):
    """Validate an external camera–LiDAR backend's BEV feature interface.

    The returned tensor is always ``[B,C,H,W]`` in the metric geometry stated
    by ``output_spec``. Backend-specific preprocessing, calibration handling,
    and checkpoint loading remain outside this package.
    """

    def __init__(self, backend: Callable[[Mapping[str, Any]], Tensor | Mapping[str, Tensor]], output_spec: BEVSpec) -> None:
        super().__init__()
        self.backend = backend
        self.output_spec = output_spec

    def forward(self, batch: Mapping[str, Any]) -> Tensor:
        """Return validated BEV features from an external backend batch."""

        produced = self.backend(batch)
        if isinstance(produced, Mapping):
            if "bev" not in produced:
                raise KeyError("external backend mapping must contain a 'bev' tensor")
            produced = produced["bev"]
        return validate_bev(produced, self.output_spec)
