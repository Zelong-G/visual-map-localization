#!/usr/bin/env python3
"""Run a dataset-free SE(2) posterior and temporal-fusion demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from temporal_maploc.evaluation.visualization import plot_pose_heatmaps
from temporal_maploc.geometry.se2 import wrap_angle
from temporal_maploc.models.pose_solver import PoseGrid
from temporal_maploc.temporal.fusion import fuse_posteriors
from temporal_maploc.temporal.pose_propagation import propagate_pose_posterior


def gaussian_posterior(grid: PoseGrid, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Create a normalized synthetic likelihood over a pose grid."""

    offsets = grid.offsets
    delta = offsets - center
    delta[:, 2] = torch.as_tensor(wrap_angle(delta[:, 2]))
    return torch.softmax(-0.5 * (delta / scale).square().sum(dim=-1), dim=-1).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional before/after PNG output path.")
    args = parser.parse_args()

    grid = PoseGrid.from_ranges(
        x_m=(-4.0, 4.0, 0.25), y_m=(-3.0, 3.0, 0.25), yaw_deg=(-12.0, 12.0, 2.0)
    )
    previous = gaussian_posterior(grid, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.25, 0.25, 0.03]))
    observation = gaussian_posterior(grid, torch.tensor([1.25, -0.20, 0.04]), torch.tensor([0.65, 0.65, 0.10]))
    initial = torch.zeros(1, 3)
    motion_mean = torch.tensor([[1.0, 0.0, 0.02]])
    motion_covariance = torch.diag_embed(torch.tensor([[0.20**2, 0.18**2, 0.04**2]]))
    propagated = propagate_pose_posterior(
        previous,
        grid=grid,
        previous_initial_pose=initial,
        current_initial_pose=initial,
        motion_mean=motion_mean,
        motion_covariance=motion_covariance,
        covariance_floor=(0.0, 0.0, 0.0),
    )
    fused = fuse_posteriors(observation, propagated.prior, prior_weight=0.85)
    offsets = grid.offsets
    estimate = fused @ offsets
    estimate[0, 2] = torch.atan2(fused @ torch.sin(offsets[:, 2]), fused @ torch.cos(offsets[:, 2]))

    print("initial pose correction [x_m, y_m, yaw_rad]: [0.000, 0.000, 0.000]")
    print("motion prior [dx_m, dy_m, dyaw_rad]: [1.000, 0.000, 0.020]")
    print("predicted correction [x_m, y_m, yaw_rad]: " + str([round(float(value), 4) for value in estimate[0]]))
    print(f"retained motion-prior mass: {float(propagated.retained_mass[0]):.6f}")
    if args.output is not None:
        path = plot_pose_heatmaps(observation[0], fused[0], grid, args.output)
        print(f"saved visualization: {path}")


if __name__ == "__main__":
    main()
