"""Tests for RL scaffolding (GRPO/RLOO/PPO math)."""
from __future__ import annotations

from triton_change.training.grpo import compute_grpo_advantages
from triton_change.training.ppo import ppo_clipped_loss
from triton_change.training.rloo import compute_rloo_advantages


def test_grpo_zero_mean_advantages():
    batch = compute_grpo_advantages([1.0, 2.0, 3.0])
    assert abs(sum(batch.advantages)) < 1e-6
    assert batch.baseline == 2.0


def test_rloo_leave_one_out():
    batch = compute_rloo_advantages([1.0, 3.0])
    assert batch.advantages[0] == 1.0 - 3.0
    assert batch.advantages[1] == 3.0 - 1.0


def test_ppo_clipped_loss():
    loss = ppo_clipped_loss(1.5, 1.0, clip_eps=0.2)
    assert loss == -1.2  # clipped at 1.2
