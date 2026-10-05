"""End-to-end causal temporal fusion around a single-frame localizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from temporal_maploc.maps.vector_map import VectorMapBatch
from temporal_maploc.models.localizer import LocalizerOutput, SingleFrameLocalizer

from .fusion import fuse_posteriors
from .motion_model import RelativeMotion
from .pose_propagation import PropagationResult, propagate_pose_posterior
from .posterior import PosePosterior


@dataclass(frozen=True)
class TemporalOutput:
    """Observation, optional motion prior, and fused current pose posterior."""

    single_frame: LocalizerOutput
    posterior: PosePosterior
    propagation: PropagationResult | None


class TemporalLocalizer(nn.Module):
    """Fuse a localizer likelihood with a propagated preceding posterior."""

    def __init__(
        self,
        single_frame: SingleFrameLocalizer,
        *,
        prior_weight: float = 0.85,
        minimum_prior_weight: float = 0.05,
        covariance_floor: tuple[float, float, float] = (0.05, 0.05, 0.008),
    ) -> None:
        super().__init__()
        if not 0.0 <= prior_weight <= 1.0:
            raise ValueError("prior_weight must be in [0,1]")
        self.single_frame = single_frame
        self.prior_weight = float(prior_weight)
        self.minimum_prior_weight = float(minimum_prior_weight)
        self.covariance_floor = covariance_floor

    def forward(
        self,
        bev: Tensor,
        vector_map: VectorMapBatch,
        *,
        current_initial_pose: Tensor,
        previous: PosePosterior | None = None,
        previous_initial_pose: Tensor | None = None,
        motion: RelativeMotion | None = None,
    ) -> TemporalOutput:
        """Run one causal temporal localization step.

        At sequence start, omit ``previous``, ``previous_initial_pose``, and
        ``motion``; the single-frame posterior is returned unchanged. On later
        frames all three must be supplied with a common batch size.
        """

        single = self.single_frame(bev, vector_map)
        observation = single.pose.probabilities
        grid = single.pose.grid
        if previous is None and previous_initial_pose is None and motion is None:
            return TemporalOutput(single, PosePosterior(observation, grid), None)
        if previous is None or previous_initial_pose is None or motion is None:
            raise ValueError("previous posterior, previous initial pose, and motion must be supplied together")
        if previous.grid.shape != grid.shape:
            raise ValueError("previous posterior grid must equal current localizer grid")
        propagation = propagate_pose_posterior(
            previous.probabilities,
            grid=grid,
            previous_initial_pose=previous_initial_pose,
            current_initial_pose=current_initial_pose,
            motion_mean=motion.mean,
            motion_covariance=motion.covariance,
            covariance_floor=self.covariance_floor,
        )
        weight = torch.as_tensor(self.prior_weight, device=observation.device, dtype=observation.dtype) * motion.reliability.to(observation)
        fused = fuse_posteriors(
            observation,
            propagation.prior,
            prior_weight=weight,
            minimum_prior_weight=self.minimum_prior_weight,
        )
        return TemporalOutput(single, PosePosterior(fused, grid), propagation)
