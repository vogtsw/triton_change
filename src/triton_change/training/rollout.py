"""RL rollout helpers (Phase 6 scaffolding)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from triton_change.agent.langgraph_app import run_rollout
from triton_change.agent.policy import PolicyBase
from triton_change.agent.trajectory import write_trajectory_jsonl


def rollout_tasks(
    task_dirs: Iterable[Path | str],
    policy_factory,
    *,
    output_jsonl: Path | None = None,
    device: str = "cpu",
    use_langgraph: bool = True,
    **runner_kw: Any,
) -> list[dict[str, Any]]:
    """Run rollouts; policy_factory(task_dir) -> PolicyBase."""
    trajectories: list[dict[str, Any]] = []
    for td in task_dirs:
        td = Path(td)
        policy = policy_factory(td)
        result = run_rollout(td, policy, device=device, use_langgraph=use_langgraph, **runner_kw)
        traj = result.to_dict()
        trajectories.append(traj)
        if output_jsonl:
            write_trajectory_jsonl([traj], output_jsonl)
    return trajectories


def load_rewards(trajectories: list[dict[str, Any]]) -> list[float]:
    return [float(t.get("final_reward", 0.0)) for t in trajectories]
