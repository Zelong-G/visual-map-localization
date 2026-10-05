"""Minimal padded vector-map segment representation.

Each element is a directed or undirected segment ``[x1, y1, x2, y2]`` in the
initial ego frame. Polygon boundaries can be represented as their constituent
segments before constructing this batch object.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VectorMapBatch:
    """Typed vector-map segments with explicit valid-element masking.

    ``segments`` is ``[B,N,4]`` in meters, ``type_ids`` is ``[B,N]``, and
    ``mask`` is ``[B,N]``. Invalid padded entries must have mask ``False`` and
    are ignored by all localization modules.
    """

    segments: Tensor
    type_ids: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        segments = torch.as_tensor(self.segments)
        types = torch.as_tensor(self.type_ids, device=segments.device)
        mask = torch.as_tensor(self.mask, device=segments.device)
        if segments.ndim != 3 or segments.shape[-1] != 4:
            raise ValueError(f"segments must be [B,N,4], got {tuple(segments.shape)}")
        if types.shape != segments.shape[:2] or mask.shape != segments.shape[:2]:
            raise ValueError("type_ids and mask must have shape [B,N]")
        if not torch.is_floating_point(segments) or not torch.isfinite(segments).all():
            raise ValueError("segments must be finite floating-point meters")
        if mask.dtype != torch.bool:
            raise TypeError("map mask must be boolean")
        if torch.is_floating_point(types):
            if not torch.equal(types, types.long().to(types.dtype)):
                raise TypeError("type_ids must be integral")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "type_ids", types.long())
        object.__setattr__(self, "mask", mask)

    @property
    def batch_size(self) -> int:
        return int(self.segments.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.segments.shape[1])

    @property
    def midpoints(self) -> Tensor:
        """Return segment midpoints ``[B,N,2]`` in initial-ego meters."""

        return 0.5 * (self.segments[..., :2] + self.segments[..., 2:])
