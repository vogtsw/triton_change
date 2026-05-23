"""RLOO (leave-one-out) baseline (Phase 6 scaffolding)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RLOOBatch:
    rewards: list[float]
    advantages: list[float]


def compute_rloo_advantages(rewards: Sequence[float]) -> RLOOBatch:
    rs = [float(r) for r in rewards]
    n = len(rs)
    if n <= 1:
        return RLOOBatch(rs, [0.0] * n)
    total = sum(rs)
    adv = []
    for r in rs:
        baseline = (total - r) / (n - 1)
        adv.append(r - baseline)
    return RLOOBatch(rewards=rs, advantages=adv)
