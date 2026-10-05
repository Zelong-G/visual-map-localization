"""Causal transport of discrete pose posteriors under relative SE(2) motion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch
import torch.nn.functional as functional
from torch import Tensor

from temporal_maploc.geometry.se2 import between, compose
from temporal_maploc.models.pose_solver import PoseGrid

from .posterior import normalize_probabilities


@dataclass(frozen=True)
class PropagationResult:
    """Motion transport result before and after normalized uncertainty spread."""

    prior: Tensor
    transported: Tensor
    retained_mass: Tensor
    propagated_offsets: Tensor


def propagate_candidate_offsets(
    grid: PoseGrid,
    previous_initial_pose: Tensor,
    current_initial_pose: Tensor,
    motion_mean: Tensor,
) -> Tensor:
    """Map prior-grid corrections into the current initial-pose frame.

    Inputs are pose tensors ``[B,3]``. ``motion_mean`` is a relative pose in
    the previous ego frame. Returns transformed corrections ``[B,K,3]`` in
    meters/radians for the current grid's coordinate convention.
    """

    previous = torch.as_tensor(previous_initial_pose)
    current = torch.as_tensor(current_initial_pose, device=previous.device, dtype=previous.dtype)
    motion = torch.as_tensor(motion_mean, device=previous.device, dtype=previous.dtype)
    if previous.ndim != 2 or previous.shape[-1] != 3 or current.shape != previous.shape or motion.shape != previous.shape:
        raise ValueError("previous initial pose, current initial pose, and motion must all be [B,3]")
    candidates = grid.offsets.to(previous).unsqueeze(0).expand(previous.shape[0], -1, -1)
    previous_actual = compose(previous.unsqueeze(1), candidates)
    current_actual = compose(previous_actual, motion.unsqueeze(1))
    return between(current.unsqueeze(1), current_actual)


def _uniform_axis_coordinate(values: Tensor, axis: Tensor) -> tuple[Tensor, Tensor]:
    if axis.numel() == 1:
        return torch.zeros_like(values, dtype=torch.long), torch.zeros_like(values)
    step = axis[1] - axis[0]
    raw = (values - axis[0]) / step
    lower = torch.floor(raw).long()
    fraction = raw - lower.to(raw.dtype)
    return lower, fraction


def trilinear_forward_splat(probabilities: Tensor, destinations: Tensor, grid: PoseGrid) -> Tensor:
    """Transport candidate mass with trilinear forward splatting.

    Args:
        probabilities: Normalized source posterior ``[B,K]``.
        destinations: Destination correction of each source candidate ``[B,K,3]``.
        grid: Destination candidate grid.

    Returns:
        Unnormalized destination mass ``[B,K]``. Mass outside the finite
        grid is intentionally discarded and reported by the caller.
    """

    probs = normalize_probabilities(probabilities)
    if destinations.shape != (probs.shape[0], grid.count, 3):
        raise ValueError("destinations must be [B,K,3] for the supplied grid")
    axes = tuple(axis.to(destinations) for axis in (grid.x, grid.y, grid.yaw))
    lowers, fractions = zip(*(_uniform_axis_coordinate(destinations[..., index], axis) for index, axis in enumerate(axes)))
    result = probs.new_zeros(probs.shape[0], grid.count)
    ny, nyaw = grid.shape[1:]
    for choices in product((0, 1), repeat=3):
        indices = [lower + choice for lower, choice in zip(lowers, choices)]
        weight = probs
        valid = torch.ones_like(probs, dtype=torch.bool)
        for axis_index, (choice, index) in enumerate(zip(choices, indices)):
            factor = fractions[axis_index] if choice else (1.0 - fractions[axis_index])
            weight = weight * factor
            valid &= (index >= 0) & (index < grid.shape[axis_index])
        flat_index = indices[0] * (ny * nyaw) + indices[1] * nyaw + indices[2]
        result.scatter_add_(1, flat_index.clamp(0, grid.count - 1), weight * valid.to(weight.dtype))
    return result


def _blur_one(volume: Tensor, covariance: Tensor, grid: PoseGrid) -> Tensor:
    """Apply a normalized diagonal Gaussian kernel to one pose volume."""

    variances = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(0.0)
    axes = tuple(axis.to(volume) for axis in (grid.x, grid.y, grid.yaw))
    kernels: list[Tensor] = []
    for variance, axis in zip(variances, axes):
        step = (axis[1] - axis[0]).abs() if axis.numel() > 1 else axis.new_tensor(1.0)
        sigma = variance.sqrt()
        radius = min(int(math.ceil(float((3.0 * sigma / step).detach().cpu()))), max(0, axis.numel() - 1))
        coordinate = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype) * step
        if radius == 0 or float(sigma.detach().cpu()) < 1.0e-8:
            kernel = torch.ones(1, device=volume.device, dtype=volume.dtype)
        else:
            kernel = torch.exp(-0.5 * (coordinate / sigma).square())
            kernel = kernel / kernel.sum()
        kernels.append(kernel)
    kernel3d = torch.einsum("i,j,k->ijk", kernels[0], kernels[1], kernels[2])
    padding = tuple((size - 1) // 2 for size in kernel3d.shape)
    return functional.conv3d(volume[None, None], kernel3d[None, None], padding=padding)[0, 0]


def propagate_pose_posterior(
    previous_probabilities: Tensor,
    *,
    grid: PoseGrid,
    previous_initial_pose: Tensor,
    current_initial_pose: Tensor,
    motion_mean: Tensor,
    motion_covariance: Tensor,
    covariance_floor: tuple[float, float, float] = (0.05, 0.05, 0.008),
) -> PropagationResult:
    """Propagate a discrete SE(2) posterior with motion and uncertainty.

    Args:
        previous_probabilities: Prior-frame pose probabilities ``[B,K]``.
        grid: Shared discrete pose grid; x/y are meters and yaw is radians.
        previous_initial_pose: Prior nominal map-frame pose ``[B,3]``.
        current_initial_pose: Current nominal map-frame pose ``[B,3]``.
        motion_mean: Relative vehicle motion ``[B,3]`` in previous ego frame.
        motion_covariance: Motion covariance ``[B,3,3]`` in squared units.
        covariance_floor: Minimum standard deviations for x/y/yaw.

    Returns:
        A normalized motion prior, raw transported mass, retained mass after
        uncertainty diffusion, and transformed candidate offsets.
    """

    probabilities = normalize_probabilities(previous_probabilities)
    covariance = torch.as_tensor(motion_covariance, device=probabilities.device, dtype=probabilities.dtype)
    if covariance.shape != (probabilities.shape[0], 3, 3):
        raise ValueError("motion_covariance must be [B,3,3]")
    floor = torch.as_tensor(covariance_floor, device=probabilities.device, dtype=probabilities.dtype)
    if floor.shape != (3,) or torch.any(floor < 0.0):
        raise ValueError("covariance_floor must contain three nonnegative standard deviations")
    stabilized = 0.5 * (covariance + covariance.transpose(-1, -2))
    stabilized = stabilized + torch.diag_embed(floor.square().expand(probabilities.shape[0], -1))
    destinations = propagate_candidate_offsets(grid, previous_initial_pose, current_initial_pose, motion_mean)
    transported = trilinear_forward_splat(probabilities, destinations, grid)
    blurred = torch.stack([_blur_one(grid.unflatten(transported[index]), stabilized[index], grid) for index in range(probabilities.shape[0])])
    blurred = grid.flatten(blurred)
    retained_mass = blurred.sum(dim=-1)
    uniform = torch.full_like(blurred, 1.0 / grid.count)
    prior = torch.where(
        (retained_mass > torch.finfo(blurred.dtype).eps).unsqueeze(-1),
        blurred / retained_mass.clamp_min(torch.finfo(blurred.dtype).eps).unsqueeze(-1),
        uniform,
    )
    return PropagationResult(prior, transported, retained_mass, destinations)
