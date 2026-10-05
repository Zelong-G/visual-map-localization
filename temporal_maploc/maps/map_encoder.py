"""Typed geometric encoder for vector-map query embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .vector_map import VectorMapBatch


class MapQueryEncoder(nn.Module):
    """Encode padded vector-map segments as localization queries.

    Input segments are ``[B,N,4]`` in initial-ego meters. The returned query
    tensor is ``[B,N,D]``; invalid map entries are exactly zeroed.
    """

    def __init__(self, type_count: int, hidden_dim: int, *, map_extent_m: float = 40.0) -> None:
        super().__init__()
        if type_count < 1 or hidden_dim < 1 or map_extent_m <= 0.0:
            raise ValueError("type_count, hidden_dim, and map_extent_m must be positive")
        self.type_count = int(type_count)
        self.map_extent_m = float(map_extent_m)
        self.type_embedding = nn.Embedding(type_count, hidden_dim)
        self.geometry = nn.Sequential(
            nn.Linear(11, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vector_map: VectorMapBatch) -> Tensor:
        """Return typed map queries ``[B,N,D]`` for the supplied segments."""

        invalid = vector_map.mask & ((vector_map.type_ids < 0) | (vector_map.type_ids >= self.type_count))
        if invalid.any():
            raise ValueError("valid map elements contain an out-of-range type_id")
        segment = vector_map.segments
        start, end = segment[..., :2], segment[..., 2:]
        delta = end - start
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        direction = delta / length.clamp_min(1.0e-6)
        midpoint = 0.5 * (start + end)
        geometry = torch.cat(
            (start, end, midpoint, delta, length / self.map_extent_m, direction), dim=-1
        )
        geometry = geometry / geometry.new_tensor([self.map_extent_m] * 8 + [1.0, 1.0, 1.0])
        safe_types = vector_map.type_ids.clamp(0, self.type_count - 1)
        query = self.geometry(geometry) + self.type_embedding(safe_types)
        return query.masked_fill(~vector_map.mask.unsqueeze(-1), 0.0)
