"""LangGraph wrapper around AgentRunner (Phase 6 scaffolding).

Falls back to direct AgentRunner.run() when langgraph is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from triton_change.agent.policy import PolicyBase
from triton_change.agent.runner import AgentRunResult, AgentRunner


class AgentGraphState(TypedDict, total=False):
    task_dir: str
    step_idx: int
    done: bool
    result: dict[str, Any]


def build_agent_graph(
    task_dir: Path | str,
    policy: PolicyBase,
    **runner_kw: Any,
):
    """Return a compiled LangGraph app, or None if langgraph unavailable.

    Graph topology (spec 5.3):
        reset -> observe_act_rollout -> END

    The rollout node runs the full AgentRunner loop (observe -> policy -> tools -> reward).
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    runner = AgentRunner(task_dir, policy, **runner_kw)

    def reset_node(state: AgentGraphState) -> AgentGraphState:
        runner._reset_workspace()
        return {"step_idx": 0, "done": False}

    def rollout_node(state: AgentGraphState) -> AgentGraphState:
        result = runner.run()
        return {"done": True, "result": result.to_dict(), "step_idx": result.total_steps}

    graph = StateGraph(AgentGraphState)
    graph.add_node("reset", reset_node)
    graph.add_node("rollout", rollout_node)
    graph.set_entry_point("reset")
    graph.add_edge("reset", "rollout")
    graph.add_edge("rollout", END)
    return graph.compile()


def run_rollout(
    task_dir: Path | str,
    policy: PolicyBase,
    *,
    use_langgraph: bool = True,
    **runner_kw: Any,
) -> AgentRunResult:
    """Run one agent episode; prefer LangGraph when installed."""
    if use_langgraph:
        app = build_agent_graph(task_dir, policy, **runner_kw)
        if app is not None:
            out = app.invoke({"task_dir": str(task_dir), "step_idx": 0, "done": False})
            raw = out.get("result")
            if raw:
                return _result_from_dict(raw)

    runner = AgentRunner(task_dir, policy, **runner_kw)
    return runner.run()


def _result_from_dict(d: dict[str, Any]) -> AgentRunResult:
    from triton_change.agent.runner import StepRecord

    steps = [
        StepRecord(
            step_idx=s["step_idx"],
            observation=s["observation"],
            action=s["action"],
            tool_result=s["tool_result"],
            step_reward=s["step_reward"],
            reward_breakdown=s.get("reward_breakdown", {}),
            failure_class=s.get("failure_class"),
            done=s["done"],
        )
        for s in d.get("steps", [])
    ]
    return AgentRunResult(
        task_id=d["task_id"],
        tier=d.get("tier"),
        change_types=d.get("change_types", []),
        agent_model=d.get("agent_model", ""),
        agent_run_id=d.get("agent_run_id", ""),
        git_commit_sha=d.get("git_commit_sha", ""),
        started_at=d.get("started_at", ""),
        finished_at=d.get("finished_at", ""),
        steps=steps,
        success=d.get("success", False),
        final_reward=d.get("final_reward", 0.0),
        failure_class=d.get("failure_class"),
        total_steps=d.get("total_steps", len(steps)),
        total_patch_ops=d.get("total_patch_ops", 0),
        final_output_path=d.get("final_output_path"),
        llm_call_log=d.get("llm_call_log", []),
    )
