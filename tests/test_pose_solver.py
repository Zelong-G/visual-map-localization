from __future__ import annotations

import torch

from temporal_maploc.geometry.bev_geometry import BEVGeometry
from temporal_maploc.maps.vector_map import VectorMapBatch
from temporal_maploc.models.pose_solver import PoseGrid, PoseSolver


def test_solver_returns_finite_normalized_posterior_and_expected_translation() -> None:
    geometry = BEVGeometry((-2.0, 2.0), (-2.0, 2.0))
    grid = PoseGrid.from_ranges(x_m=(-1.0, 1.0, 1.0), y_m=(0.0, 0.0, 1.0), yaw_deg=(0.0, 0.0, 1.0))
    coordinates = torch.linspace(2.0, -2.0, 9)
    evidence = torch.exp(-coordinates.square()).view(1, 1, 9, 1).expand(1, 1, 9, 9)
    bev = torch.cat((evidence, 1.0 - evidence), dim=1)
    vector_map = VectorMapBatch(
        segments=torch.tensor([[[1.0, 0.0, 1.0, 0.0]]]),
        type_ids=torch.tensor([[0]]),
        mask=torch.tensor([[True]]),
    )
    solver = PoseSolver(grid, geometry, samples_per_segment=1, candidate_chunk_size=2)
    output = solver(bev, torch.tensor([[[1.0, 0.0]]]), vector_map)
    assert torch.isfinite(output.probabilities).all()
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(1))
    assert torch.allclose(output.argmax, torch.tensor([[1.0, 0.0, 0.0]]), atol=1.0e-4)
