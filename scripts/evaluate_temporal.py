#!/usr/bin/env python3
"""Run a data-free temporal-posterior contract evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from temporal_maploc.config import load_yaml
from temporal_maploc.models.pose_solver import PoseGrid
from temporal_maploc.temporal.fusion import fuse_posteriors
from temporal_maploc.temporal.pose_propagation import propagate_pose_posterior


def _gaussian(grid: PoseGrid, center: torch.Tensor) -> torch.Tensor:
    offsets = grid.offsets
    scale = torch.tensor([0.5, 0.5, 0.08])
    return torch.softmax(-0.5 * ((offsets - center) / scale).square().sum(dim=-1), dim=-1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Temporal YAML configuration.")
    parser.add_argument("--synthetic", action="store_true", help="Run the included data-free sequence contract.")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is available until a releasable sequence-loader contract is supplied.")
    config = load_yaml(args.config)
    source = Path(config["single_frame_config"])
    if not source.is_absolute():
        source = args.config.parent / source
    single = load_yaml(source)
    grid_cfg = single["pose_grid"]
    grid = PoseGrid.from_ranges(
        x_m=tuple(grid_cfg["x_m"].values()),
        y_m=tuple(grid_cfg["y_m"].values()),
        yaw_deg=tuple(grid_cfg["yaw_deg"].values()),
    )
    previous = _gaussian(grid, torch.tensor([0.0, 0.0, 0.0]))
    observation = _gaussian(grid, torch.tensor([1.0, 0.0, 0.0]))
    propagation = propagate_pose_posterior(
        previous,
        grid=grid,
        previous_initial_pose=torch.zeros(1, 3),
        current_initial_pose=torch.zeros(1, 3),
        motion_mean=torch.tensor([[1.0, 0.0, 0.0]]),
        motion_covariance=torch.diag_embed(torch.tensor([[0.15**2, 0.15**2, 0.03**2]])),
        covariance_floor=tuple(config["temporal"]["covariance_floor"]),
    )
    fused = fuse_posteriors(observation, propagation.prior, prior_weight=float(config["temporal"]["prior_weight"]))
    assert torch.isfinite(fused).all() and torch.allclose(fused.sum(dim=-1), torch.ones(1), atol=1.0e-6)
    print({"candidate_count": grid.count, "retained_mass": float(propagation.retained_mass[0]), "posterior_mass": float(fused.sum())})


if __name__ == "__main__":
    main()
