"""Compact cross-modal BEV-to-map query decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from temporal_maploc.geometry.bev_geometry import BEVGeometry, sample_bev
from temporal_maploc.maps.vector_map import VectorMapBatch


class CrossModalDecoder(nn.Module):
    """Refine map queries with pooled BEV context and self-attention.

    Args:
        hidden_dim: Shared BEV/map embedding dimension.
        heads: Number of attention heads.
        layers: Number of residual decoder layers.
        geometry: Metric convention for ``bev_feature``.

    The input BEV tensor is ``[B,D,H,W]`` and map queries are ``[B,N,D]``.
    Segment midpoints are sampled in initial-ego meters. The output preserves
    ``[B,N,D]`` and padded map entries remain zero.
    """

    def __init__(self, hidden_dim: int, heads: int, layers: int, geometry: BEVGeometry) -> None:
        super().__init__()
        if hidden_dim < 1 or heads < 1 or layers < 1 or hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by positive heads; layers must be positive")
        self.geometry = geometry
        self.context_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, bev_feature: Tensor, map_queries: Tensor, vector_map: VectorMapBatch) -> Tensor:
        """Return cross-modal map queries for BEV ``[B,D,H,W]``.

        The pooled context stabilizes attention cost for high-resolution BEV
        inputs, while midpoint sampling preserves a direct metric connection
        between vector elements and the BEV coordinate frame.
        """

        if bev_feature.ndim != 4 or map_queries.ndim != 3:
            raise ValueError("bev_feature must be [B,D,H,W] and map_queries must be [B,N,D]")
        if bev_feature.shape[0] != vector_map.batch_size or tuple(map_queries.shape[:2]) != tuple(vector_map.mask.shape):
            raise ValueError("BEV, map queries, and map batch must share [B,N]")
        if bev_feature.shape[1] != map_queries.shape[-1]:
            raise ValueError("BEV and map query embeddings must share hidden_dim")
        local_context = sample_bev(bev_feature, vector_map.midpoints, self.geometry)
        pooled = functional.adaptive_avg_pool2d(bev_feature, output_size=(16, 16)).flatten(2).transpose(1, 2)
        query = map_queries + self.context_projection(local_context)
        # TransformerEncoder accepts a source-padding mask; all-false maps are
        # handled by temporarily enabling one zero query and masking it later.
        mask = vector_map.mask
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        query = query.masked_fill(~safe_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            query = layer(query, src_key_padding_mask=~safe_mask)
            query = query + 0.1 * torch.tanh(query.mean(dim=1, keepdim=True))
        # Pooled BEV context is intentionally a residual rather than a second
        # expensive cross-attention implementation.
        pooled_context = pooled.mean(dim=1, keepdim=True)
        query = self.output_norm(query + pooled_context)
        return query.masked_fill(~mask.unsqueeze(-1), 0.0)
