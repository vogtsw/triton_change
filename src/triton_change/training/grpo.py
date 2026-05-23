"""GRPO advantage computation (Phase 6 scaffolding — no trainer backend)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class GRPOBatch:
    rewards: list[float]
    advantages: list[float]
    baseline: float


def compute_grpo_advantages(rewards: Sequence[float], *, eps: float = 1e-8) -> GRPOBatch:
    """Group-relative advantages: each reward minus group mean."""
    rs = [float(r) for r in rewards]
    if not rs:
        return GRPOBatch([], [], 0.0)
    baseline = sum(rs) / len(rs)
    adv = [r - baseline for r in rs]
    return GRPOBatch(rewards=rs, advantages=adv, baseline=baseline)
