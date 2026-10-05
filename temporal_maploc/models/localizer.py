"""End-to-end single-frame BEV-to-vector-map localization module."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from temporal_maploc.bev.bev_encoder import BEVEncoder
from temporal_maploc.maps.map_encoder import MapQueryEncoder
from temporal_maploc.maps.vector_map import VectorMapBatch

from .cross_modal_decoder import CrossModalDecoder
from .pose_solver import PoseSolver, PoseSolverOutput


@dataclass(frozen=True)
class LocalizerOutput:
    """Inspectable outputs from one single-frame localization pass."""

    bev_features: Tensor
    map_queries: Tensor
    pose: PoseSolverOutput


class SingleFrameLocalizer(nn.Module):
    """Compose BEV encoding, map encoding, matching, and SE(2) scoring."""

    def __init__(
        self,
        bev_encoder: BEVEncoder,
        map_encoder: MapQueryEncoder,
        decoder: CrossModalDecoder,
        pose_solver: PoseSolver,
    ) -> None:
        super().__init__()
        self.bev_encoder = bev_encoder
        self.map_encoder = map_encoder
        self.decoder = decoder
        self.pose_solver = pose_solver

    def forward(self, bev: Tensor, vector_map: VectorMapBatch) -> LocalizerOutput:
        """Infer an SE(2) correction posterior from BEV and vector-map inputs."""

        bev_features = self.bev_encoder(bev)
        map_queries = self.map_encoder(vector_map)
        map_queries = self.decoder(bev_features, map_queries, vector_map)
        pose = self.pose_solver(bev_features, map_queries, vector_map)
        return LocalizerOutput(bev_features, map_queries, pose)
