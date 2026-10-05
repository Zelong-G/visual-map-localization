"""Optional Matplotlib visualizations for 2-D marginalized pose posteriors."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from temporal_maploc.models.pose_solver import PoseGrid


def plot_pose_heatmaps(
    observation: Tensor,
    fused: Tensor,
    grid: PoseGrid,
    output: str | Path,
) -> Path:
    """Save before/after XY posterior heatmaps marginalized over yaw.

    Both posterior inputs are flattened ``[K]`` vectors over ``grid``. Axes
    are labelled in initial-ego meters; no dataset image or map is required.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - dependency is optional
        raise RuntimeError("install TemporalMapLoc[visualization] to create plots") from error
    before = grid.unflatten(torch.as_tensor(observation)).sum(dim=-1).detach().cpu()
    after = grid.unflatten(torch.as_tensor(fused)).sum(dim=-1).detach().cpu()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    extent = [float(grid.yaw[0]), float(grid.yaw[-1]), float(grid.x[0]), float(grid.x[-1])]
    # Use a conventional x/y visualization independent of yaw by plotting
    # y horizontally and x vertically.
    extent = [float(grid.y[0]), float(grid.y[-1]), float(grid.x[0]), float(grid.x[-1])]
    for axis, image, title in zip(axes, (before, after), ("Current observation", "Temporal posterior")):
        artist = axis.imshow(image, origin="lower", aspect="auto", extent=extent)
        axis.set_title(title)
        axis.set_xlabel("y / m (left)")
        axis.set_ylabel("x / m (forward)")
        figure.colorbar(artist, ax=axis, label="probability")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
