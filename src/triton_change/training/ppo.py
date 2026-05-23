"""PPO-style clipped objective helper (Phase 6 scaffolding — math only)."""
from __future__ import annotations

import math


def ppo_clipped_loss(
    ratio: float,
    advantage: float,
    *,
    clip_eps: float = 0.2,
) -> float:
    """Standard single-sample PPO surrogate (for unit tests / plumbing)."""
    unclipped = ratio * advantage
    clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio)) * advantage
    return -min(unclipped, clipped)


def log_ratio_from_rewards(old_logp: float, new_logp: float) -> float:
    return new_logp - old_logp
