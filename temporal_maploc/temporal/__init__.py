"""Causal motion models and probabilistic temporal localization."""

from .fusion import fuse_posteriors
from .motion_model import RelativeMotion, integrate_planar_twist
from .pose_propagation import propagate_pose_posterior
from .posterior import PosePosterior, posterior_expectation
from .temporal_localizer import TemporalLocalizer, TemporalOutput

__all__ = [
    "PosePosterior",
    "RelativeMotion",
    "TemporalLocalizer",
    "TemporalOutput",
    "fuse_posteriors",
    "integrate_planar_twist",
    "posterior_expectation",
    "propagate_pose_posterior",
]
