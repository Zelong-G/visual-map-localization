"""Discrete SE(2) pose grid and vector-map-aware exhaustive pose scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import product

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from temporal_maploc.geometry.bev_geometry import BEVGeometry
from temporal_maploc.geometry.se2 import wrap_angle
from temporal_maploc.maps.vector_map import VectorMapBatch


def _axis(minimum: float, maximum: float, step: float, *, device: torch.device | None = None) -> Tensor:
    if not math.isfinite(minimum) or not math.isfinite(maximum) or not math.isfinite(step):
        raise ValueError("pose-grid values must be finite")
    if step <= 0.0 or maximum < minimum:
        raise ValueError("pose-grid step must be positive and max must be at least min")
    values = torch.arange(minimum, maximum + step * 0.5, step, dtype=torch.float32, device=device)
    return values[values <= maximum + max(1.0, abs(maximum)) * 1.0e-6]


@dataclass(frozen=True)
class PoseGrid:
    """Cartesian grid of correction offsets in initial-ego coordinates.

    ``x`` and ``y`` are meters; ``yaw`` is radians. The flattened candidate
    order is `(x, y, yaw)` with yaw varying fastest.
    """

    x: Tensor
    y: Tensor
    yaw: Tensor

    def __post_init__(self) -> None:
        axes = tuple(torch.as_tensor(axis, dtype=torch.float32) for axis in (self.x, self.y, self.yaw))
        if any(axis.ndim != 1 or axis.numel() == 0 for axis in axes):
            raise ValueError("each pose-grid axis must be a nonempty one-dimensional tensor")
        if any(not torch.isfinite(axis).all() or (axis.numel() > 1 and not torch.all(axis[1:] > axis[:-1])) for axis in axes):
            raise ValueError("pose-grid axes must be finite and strictly increasing")
        object.__setattr__(self, "x", axes[0])
        object.__setattr__(self, "y", axes[1])
        object.__setattr__(self, "yaw", axes[2])

    @classmethod
    def from_ranges(
        cls,
        *,
        x_m: tuple[float, float, float],
        y_m: tuple[float, float, float],
        yaw_deg: tuple[float, float, float],
    ) -> "PoseGrid":
        """Create an inclusive grid; yaw input is in degrees."""

        return cls(
            _axis(*x_m),
            _axis(*y_m),
            _axis(*(math.radians(value) for value in yaw_deg)),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.x.numel()), int(self.y.numel()), int(self.yaw.numel()))

    @property
    def count(self) -> int:
        return int(self.x.numel() * self.y.numel() * self.yaw.numel())

    @property
    def offsets(self) -> Tensor:
        mesh = torch.meshgrid(self.x, self.y, self.yaw, indexing="ij")
        return torch.stack(mesh, dim=-1).reshape(-1, 3)

    def to(self, device: torch.device | str) -> "PoseGrid":
        return PoseGrid(self.x.to(device), self.y.to(device), self.yaw.to(device))

    def flatten(self, volume: Tensor) -> Tensor:
        if tuple(volume.shape[-3:]) != self.shape:
            raise ValueError(f"volume must end in pose-grid shape {self.shape}")
        return volume.reshape(*volume.shape[:-3], self.count)

    def unflatten(self, probabilities: Tensor) -> Tensor:
        if probabilities.shape[-1] != self.count:
            raise ValueError(f"posterior must end in {self.count} candidates")
        return probabilities.reshape(*probabilities.shape[:-1], *self.shape)


@dataclass(frozen=True)
class PoseSolverOutput:
    """Scores and posterior statistics from discrete pose matching."""

    scores: Tensor
    log_probabilities: Tensor
    probabilities: Tensor
    estimate: Tensor
    argmax: Tensor
    covariance: Tensor
    grid: PoseGrid


class PoseSolver(nn.Module):
    """Score pose hypotheses by matching BEV samples to vector-map queries.

    Map segments are expressed in the initial ego frame. For each candidate
    correction, their points are transformed into the candidate ego frame,
    bilinearly sampled in a BEV feature tensor ``[B,D,H,W]``, and compared to
    queries ``[B,N,D]``. Candidate evaluation is chunked to control memory.
    """

    def __init__(
        self,
        grid: PoseGrid,
        geometry: BEVGeometry,
        *,
        samples_per_segment: int = 5,
        candidate_chunk_size: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if samples_per_segment < 1 or candidate_chunk_size < 1 or temperature <= 0.0:
            raise ValueError("samples_per_segment, chunk size, and temperature must be positive")
        self.grid = grid
        self.geometry = geometry
        self.samples_per_segment = int(samples_per_segment)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.register_buffer("candidate_offsets", grid.offsets, persistent=False)
        self.register_buffer("temperature", torch.tensor(float(temperature)), persistent=False)

    def _segment_points(self, segments: Tensor) -> Tensor:
        weights = torch.linspace(0.0, 1.0, self.samples_per_segment, device=segments.device, dtype=segments.dtype)
        start, end = segments[..., :2], segments[..., 2:]
        return start.unsqueeze(-2) + (end - start).unsqueeze(-2) * weights.view(1, 1, -1, 1)

    @staticmethod
    def _to_candidate_ego(points: Tensor, candidates: Tensor) -> Tensor:
        """Express initial-ego points in each candidate ego frame."""

        translated = points[:, None] - candidates[None, :, None, None, :2]
        cosine = torch.cos(candidates[:, 2])[None, :, None, None]
        sine = torch.sin(candidates[:, 2])[None, :, None, None]
        x = cosine * translated[..., 0] + sine * translated[..., 1]
        y = -sine * translated[..., 0] + cosine * translated[..., 1]
        return torch.stack((x, y), dim=-1)

    def score(self, bev_feature: Tensor, map_queries: Tensor, vector_map: VectorMapBatch) -> Tensor:
        """Return unnormalized candidate scores of shape ``[B,K]``."""

        if bev_feature.ndim != 4 or map_queries.ndim != 3:
            raise ValueError("BEV must be [B,D,H,W] and map queries must be [B,N,D]")
        batch, channels = bev_feature.shape[:2]
        if map_queries.shape != (batch, vector_map.element_count, channels):
            raise ValueError("map queries must be [B,N,D] aligned with BEV channels and map elements")
        points = self._segment_points(vector_map.segments)
        normalized_queries = functional.normalize(map_queries, dim=-1, eps=1.0e-6)
        scores: list[Tensor] = []
        offsets = self.candidate_offsets.to(device=bev_feature.device, dtype=bev_feature.dtype)
        for start in range(0, offsets.shape[0], self.candidate_chunk_size):
            candidates = offsets[start : start + self.candidate_chunk_size]
            candidate_points = self._to_candidate_ego(points, candidates)
            b, k, n, samples, _ = candidate_points.shape
            # Grid-sample can evaluate every candidate location against the
            # same source feature map in one call. Packing candidate points in
            # its output-height axis avoids materializing K copies of a large
            # BEV tensor.
            grid = self.geometry.metric_to_grid(candidate_points).reshape(b, k * n * samples, 1, 2)
            sampled = functional.grid_sample(bev_feature, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, k, n, samples, channels)
            sampled = functional.normalize(sampled, dim=-1, eps=1.0e-6)
            segment_scores = (sampled * normalized_queries[:, None, :, None, :]).sum(dim=-1).mean(dim=-1)
            weights = vector_map.mask[:, None].to(dtype=segment_scores.dtype)
            scores.append((segment_scores * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0))
        return torch.cat(scores, dim=1)

    def forward(self, bev_feature: Tensor, map_queries: Tensor, vector_map: VectorMapBatch) -> PoseSolverOutput:
        """Compute score volume, normalized posterior, and SE(2) estimates."""

        scores = self.score(bev_feature, map_queries, vector_map)
        log_probabilities = functional.log_softmax(scores / self.temperature.to(scores), dim=-1)
        probabilities = log_probabilities.exp()
        offsets = self.candidate_offsets.to(probabilities)
        xy = probabilities @ offsets[:, :2]
        sine = probabilities @ torch.sin(offsets[:, 2])
        cosine = probabilities @ torch.cos(offsets[:, 2])
        yaw = torch.atan2(sine, cosine)
        estimate = torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)
        delta = offsets.unsqueeze(0) - estimate.unsqueeze(1)
        delta[..., 2] = torch.as_tensor(wrap_angle(delta[..., 2]))
        covariance = torch.einsum("bk,bki,bkj->bij", probabilities, delta, delta)
        argmax = offsets.index_select(0, probabilities.argmax(dim=-1))
        return PoseSolverOutput(scores, log_probabilities, probabilities, estimate, argmax, covariance, self.grid.to(probabilities.device))
