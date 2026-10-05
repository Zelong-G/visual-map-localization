"""Numerically stable fusion of observation and motion-prior posteriors."""

from __future__ import annotations

import torch
from torch import Tensor

from .posterior import normalize_probabilities


def fuse_posteriors(
    observation: Tensor,
    motion_prior: Tensor,
    *,
    prior_weight: Tensor | float,
    minimum_prior_weight: float = 0.05,
    eps: float = 1.0e-12,
) -> Tensor:
    """Fuse current likelihood and motion prior with a safe product of experts.

    Args:
        observation: Current-frame posterior/likelihood ``[B,K]``.
        motion_prior: Transported posterior ``[B,K]``.
        prior_weight: Motion reliability in `[0,1]`, scalar or ``[B]``.
        minimum_prior_weight: Lower clamp that prevents an exact zero branch.

    Returns:
        A finite normalized posterior ``[B,K]``. The safe prior mixes the
        motion distribution with uniform mass before log-space fusion.
    """

    obs = normalize_probabilities(observation)
    prior = normalize_probabilities(motion_prior)
    if prior.shape != obs.shape:
        raise ValueError("observation and motion_prior must have identical [B,K] shape")
    weight = torch.as_tensor(prior_weight, device=obs.device, dtype=obs.dtype)
    if weight.ndim == 0:
        weight = weight.expand(obs.shape[0])
    if weight.shape != (obs.shape[0],):
        raise ValueError("prior_weight must be scalar or [B]")
    if not 0.0 <= minimum_prior_weight <= 1.0:
        raise ValueError("minimum_prior_weight must be in [0,1]")
    weight = weight.clamp(minimum_prior_weight, 1.0)
    uniform = torch.full_like(prior, 1.0 / prior.shape[-1])
    safe_prior = weight[:, None] * prior + (1.0 - weight[:, None]) * uniform
    logits = obs.clamp_min(eps).log() + safe_prior.clamp_min(eps).log()
    return torch.softmax(logits, dim=-1)
