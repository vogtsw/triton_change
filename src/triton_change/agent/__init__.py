"""LangGraph-style multi-step agent for the Triton kernel migration task.

Design overview (see spec section 5.3):

    observe -> propose+apply -> static_check -> correctness_check -> reward
    feedback -> next observation -> ... (until success / max_steps / finalize)

The runner is a plain Python loop so it is fully testable and trivially
wrappable in LangGraph (or any other graph framework) later. The policy is
an interface with three concrete implementations:

    DeepSeekPolicy   uses DeepSeek V3 via OpenAI-compatible API
    OraclePolicy     replays oracle/patch_ops.json (sanity check)
    MockPolicy       replays a hard-coded action sequence (testing)

Trajectories follow `schemas/trajectory_schema.json`.
"""
from triton_change.agent.observation import (
    CodeSummary,
    Observation,
    extract_code_summary,
    failure_hint,
)
from triton_change.agent.policy import (
    DeepSeekPolicy,
    MockPolicy,
    OraclePolicy,
    PolicyBase,
)
from triton_change.agent.runner import AgentRunner, AgentRunResult
from triton_change.agent.tools import ToolResult
from triton_change.agent.trajectory import write_trajectory_jsonl

__all__ = [
    "CodeSummary",
    "Observation",
    "extract_code_summary",
    "failure_hint",
    "DeepSeekPolicy",
    "MockPolicy",
    "OraclePolicy",
    "PolicyBase",
    "AgentRunner",
    "AgentRunResult",
    "ToolResult",
    "write_trajectory_jsonl",
]
