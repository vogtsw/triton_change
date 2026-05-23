"""AgentRunner — the core observe -> act -> tool -> reward loop.

Pure Python. No LangGraph dependency. The same loop can be wrapped in a
LangGraph state graph later (each node = one of: observe / propose+apply /
static / correctness / reward / finalize) without changing any of the
underlying tool logic.

Per-step flow:

    1. Build observation from current state.
    2. Ask policy for next action.
    3. Dispatch action -> ToolResult.
    4. If action was apply_patch_ops, automatically chain:
         run_static_check
         (if static passes) run_correctness_check
       so the policy doesn't have to micromanage these.
    5. Compute step_reward (using triton_change.reward).
    6. Append step record. Check termination.

Termination conditions (per spec):
- success (correctness passed and policy issued finalize) -> done
- max_steps reached -> forced finalize
- max_patch_attempts exceeded -> forced finalize
- repeated_same_error 3+ times -> forced finalize
- policy returns finalize -> done
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_change.agent.observation import (
    Observation,
    error_signature,
    extract_code_summary,
    failure_hint,
    load_diff_summary,
)
from triton_change.agent.policy import PolicyBase
from triton_change.agent.tools import (
    apply_patch_ops_tool,
    benchmark_tool,
    correctness_check_tool,
    finalize_tool,
    inspect_code_region_tool,
    static_check_tool,
    ToolResult,
)
from triton_change.reward import compute_reward


__all__ = ["AgentRunner", "AgentRunResult", "StepRecord", "VALID_AGENT_TOOLS"]


# Mirrors the trajectory_schema's action.tool enum.
VALID_AGENT_TOOLS = {
    "inspect_code_region",
    "propose_patch",
    "apply_patch_ops",
    "run_static_check",
    "run_correctness_check",
    "run_benchmark",
    "finalize",
}


@dataclass
class StepRecord:
    step_idx: int
    observation: dict[str, Any]
    action: dict[str, Any]
    tool_result: dict[str, Any]
    step_reward: float
    reward_breakdown: dict[str, float]
    failure_class: str | None
    done: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "observation": self.observation,
            "action": self.action,
            "tool_result": self.tool_result,
            "step_reward": self.step_reward,
            "reward_breakdown": self.reward_breakdown,
            "failure_class": self.failure_class,
            "done": self.done,
        }


@dataclass
class AgentRunResult:
    task_id: str
    tier: str | None
    change_types: list[str]
    agent_model: str
    agent_run_id: str
    git_commit_sha: str
    started_at: str
    finished_at: str
    steps: list[StepRecord]
    success: bool
    final_reward: float
    failure_class: str | None
    total_steps: int
    total_patch_ops: int
    final_output_path: str | None
    llm_call_log: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "change_types": self.change_types,
            "agent_model": self.agent_model,
            "agent_run_id": self.agent_run_id,
            "git_commit_sha": self.git_commit_sha,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "final_reward": self.final_reward,
            "failure_class": self.failure_class,
            "total_steps": self.total_steps,
            "total_patch_ops": self.total_patch_ops,
            "final_output_path": self.final_output_path,
            "llm_call_log": self.llm_call_log,
        }


# ---------- Runner ----------


class AgentRunner:
    def __init__(
        self,
        task_dir: Path | str,
        policy: PolicyBase,
        *,
        max_steps: int = 8,
        max_patch_attempts: int = 5,
        max_repeated_errors: int = 3,
        device: str = "cpu",
        correctness_timeout: float = 60.0,
        candidate_filename: str = "candidate_model_triton.py",
    ):
        self.task_dir = Path(task_dir).resolve()
        self.policy = policy
        self.max_steps = max_steps
        self.max_patch_attempts = max_patch_attempts
        self.max_repeated_errors = max_repeated_errors
        self.device = device
        self.correctness_timeout = correctness_timeout
        self.candidate_filename = candidate_filename

    # ----- main entry point -----

    def run(self) -> AgentRunResult:
        # Clean leftover state from prior runs so the agent always starts with
        # a fresh candidate. Without this, a stale candidate from a previous
        # invocation pollutes the very first inspection step.
        self._reset_workspace()

        meta = self._read_meta()
        diff_summary = load_diff_summary(self.task_dir)
        semantic_labels = self._read_semantic_labels()

        steps: list[StepRecord] = []
        last_tool_result: ToolResult | None = None
        last_action: dict[str, Any] | None = None
        last_signature: str | None = None
        signature_streak = 0
        patch_attempts = 0

        candidate_path = self.task_dir / self.candidate_filename
        # Step 0 always starts from old_model_triton.py (no candidate yet)
        last_static: dict[str, Any] | None = None
        last_correctness: dict[str, Any] | None = None
        last_patch_ops: list[dict[str, Any]] = []
        success = False
        final_failure_class: str | None = None

        started = _now_iso()

        for step_idx in range(self.max_steps):
            # 1. Build observation
            code_src = candidate_path if candidate_path.exists() else self.task_dir / "old_model_triton.py"
            try:
                code_summary = extract_code_summary(code_src).to_dict()
            except SyntaxError:
                # Patched candidate had syntax issue; fall back to summarizing nothing.
                code_summary = {"error": "candidate has syntax error", "constants": [], "kernels": [], "functions": []}

            last_error: dict[str, Any] | None = None
            hint: str | None = None
            if last_tool_result is not None and not last_tool_result.success:
                last_error = {
                    "tool": last_tool_result.tool,
                    "failure_class": last_tool_result.failure_class,
                    "error_tail": (last_tool_result.error or "")[-2000:],
                }
                hint = failure_hint(last_tool_result.failure_class,
                                    repeated=signature_streak >= 2)

            obs = Observation(
                step_idx=step_idx,
                task_id=meta.get("task_id", self.task_dir.name),
                onnx_diff=diff_summary,
                code_summary=code_summary,
                last_action=_summarize_action(last_action) if last_action else None,
                last_error=last_error,
                hint=hint,
                remaining_steps=self.max_steps - step_idx,
                repeated_same_error=signature_streak >= 2,
            )

            # 2. Ask policy
            action = self.policy.next_action(obs)
            # Coerce invalid tools to keep the trajectory schema-valid; preserve
            # what the policy actually returned in `_original_action`.
            if not isinstance(action, dict) or action.get("tool") not in VALID_AGENT_TOOLS:
                action = {
                    "tool": "finalize",
                    "reason": f"invalid tool from policy: {action!r}"[:300],
                    "_original_action": action if isinstance(action, dict) else {"raw": str(action)},
                }
            # Bound CoT leakage: cap any 'reason' fields. The spec invariant
            # forbids storing hidden chain-of-thought in trajectories.
            action = _sanitize_action(action)
            last_action = action

            # 3. Dispatch -> tool result(s).
            tool_results: list[ToolResult] = []
            tool = action.get("tool")

            if tool == "apply_patch_ops":
                patch_attempts += 1
                ops = action.get("patch_ops") or []
                last_patch_ops = ops
                apply_res = apply_patch_ops_tool(
                    self.task_dir, ops, candidate_filename=self.candidate_filename,
                )
                tool_results.append(apply_res)

                if apply_res.success:
                    static_res = static_check_tool(candidate_path)
                    tool_results.append(static_res)
                    last_static = static_res.payload

                    if static_res.success:
                        corr_res = correctness_check_tool(
                            candidate_path, self.task_dir,
                            timeout=self.correctness_timeout, device=self.device,
                        )
                        tool_results.append(corr_res)
                        last_correctness = corr_res.payload
                        if corr_res.success:
                            success = True
                    else:
                        last_correctness = None
                else:
                    last_static = None
                    last_correctness = None

            elif tool == "inspect_code_region":
                # Inspect the current candidate if it exists; otherwise the
                # initial old_model_triton.py the agent is about to migrate.
                inspect_target = candidate_path if candidate_path.exists() else self.task_dir / "old_model_triton.py"
                tool_results.append(
                    inspect_code_region_tool(inspect_target, action.get("region", "full"))
                )
            elif tool == "run_static_check":
                static_res = static_check_tool(candidate_path)
                tool_results.append(static_res)
                last_static = static_res.payload
            elif tool == "run_correctness_check":
                corr_res = correctness_check_tool(
                    candidate_path, self.task_dir,
                    timeout=self.correctness_timeout, device=self.device,
                )
                tool_results.append(corr_res)
                last_correctness = corr_res.payload
                if corr_res.success:
                    success = True
            elif tool == "run_benchmark":
                bench_res = benchmark_tool(
                    self.task_dir, candidate_path, device=self.device,
                )
                tool_results.append(bench_res)
            elif tool == "finalize":
                tool_results.append(finalize_tool(action.get("reason", "")))
            else:
                tool_results.append(ToolResult(
                    tool=tool or "unknown", success=False,
                    payload={"raw": action},
                    error=f"unknown tool: {tool!r}",
                    failure_class="patch_apply_error",
                ))

            # The "primary" tool result for this step (for repeated-error detection)
            primary = tool_results[0]
            last_tool_result = primary if not all(r.success for r in tool_results) else tool_results[-1]
            # Use the first FAILING result for signature, or the apply_patch chain's last failing one.
            failing = next((r for r in tool_results if not r.success), None)

            # 4. Reward (only meaningful when we actually applied a patch)
            timed_out = (last_correctness or {}).get("failure_class") == "timeout"
            patch_apply_error = (
                tool == "apply_patch_ops" and not apply_res.success  # noqa
                if tool == "apply_patch_ops" else False
            )

            rb = compute_reward(
                static_result=last_static,
                correctness_result=last_correctness,
                patch_ops=last_patch_ops if tool in {"apply_patch_ops"} else [],
                semantic_labels=semantic_labels,
                timed_out=timed_out,
                repeated_same_error=signature_streak >= 2,
                patch_apply_error=patch_apply_error,
            )

            step_reward = rb.total
            reward_breakdown = dict(rb.components)
            failure_class_for_step = (
                rb.failure_class
                or (failing.failure_class if failing else None)
            )

            # 5. Repeated-error tracking
            if failing is not None:
                sig = error_signature({
                    "failure_class": failing.failure_class,
                    "error": failing.error or "",
                })
                if sig == last_signature:
                    signature_streak += 1
                else:
                    signature_streak = 1
                    last_signature = sig
            else:
                # success step — reset streak
                signature_streak = 0
                last_signature = None

            # 6. Decide done
            done_now = False
            if tool == "finalize":
                done_now = True
            elif success and tool == "apply_patch_ops":
                # Allow agent one more step to issue finalize, but if it doesn't
                # come (e.g. mock policy lacks it) we still terminate cleanly.
                pass
            if patch_attempts >= self.max_patch_attempts and not success:
                done_now = True
            if signature_streak >= self.max_repeated_errors:
                done_now = True

            steps.append(StepRecord(
                step_idx=step_idx,
                observation=obs.to_dict(),
                action=action,
                tool_result={"results": [r.to_dict() for r in tool_results]},
                step_reward=step_reward,
                reward_breakdown=reward_breakdown,
                failure_class=failure_class_for_step,
                done=done_now or success,
            ))

            if done_now:
                break
            if success and tool == "apply_patch_ops":
                # Loop one more turn to give the policy a chance to finalize cleanly.
                continue

        # ---- Final reward / failure class ----
        # We use the LAST step's failure_class so early-stage failures
        # (patch_apply_error, syntax) are preserved instead of being masked
        # by the recomputed reward (which lacks the patch_apply_error flag).
        final_rb = compute_reward(
            static_result=last_static,
            correctness_result=last_correctness,
            patch_ops=last_patch_ops,
            semantic_labels=semantic_labels,
        )
        if success:
            final_failure_class = None
            final_reward = final_rb.total
        elif steps:
            final_failure_class = steps[-1].failure_class
            # Use the last step's reward as final (already includes any penalties)
            final_reward = steps[-1].step_reward
        else:
            final_failure_class = final_rb.failure_class
            final_reward = final_rb.total

        finished = _now_iso()
        return AgentRunResult(
            task_id=meta.get("task_id", self.task_dir.name),
            tier=meta.get("tier"),
            change_types=list(meta.get("change_types", [])),
            agent_model=getattr(self.policy, "name", "unknown"),
            agent_run_id=str(uuid.uuid4()),
            git_commit_sha=_git_commit_sha(),
            started_at=started,
            finished_at=finished,
            steps=steps,
            success=success,
            final_reward=final_reward,
            failure_class=final_failure_class,
            total_steps=len(steps),
            total_patch_ops=sum(
                len(s.action.get("patch_ops", []) or [])
                for s in steps
                if s.action.get("tool") == "apply_patch_ops"
            ),
            final_output_path=str(self.task_dir / self.candidate_filename)
            if (self.task_dir / self.candidate_filename).exists() else None,
            llm_call_log=self.policy.call_log(),
        )

    # ----- helpers -----

    def _reset_workspace(self) -> None:
        import shutil
        cand = self.task_dir / self.candidate_filename
        if cand.exists():
            cand.unlink()
        sandbox = self.task_dir / "sandbox"
        if sandbox.exists():
            shutil.rmtree(sandbox, ignore_errors=True)

    def _read_meta(self) -> dict[str, Any]:
        p = self.task_dir / "meta.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _read_semantic_labels(self) -> list[dict[str, Any]]:
        p = self.task_dir / "hidden_eval" / "semantic_change_labels.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8")).get("labels", [])


# ---------- module helpers ----------


def _summarize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Compress an action for storage in the next observation. We keep tool +
    op count + first op type, but drop large payloads like full_file_replace
    new_code (which can be many KB) to keep observations lean."""
    if not action:
        return {}
    tool = action.get("tool")
    if tool != "apply_patch_ops":
        return {"tool": tool}
    ops = action.get("patch_ops") or []
    return {
        "tool": "apply_patch_ops",
        "n_ops": len(ops),
        "op_types": [op.get("operation") for op in ops[:8]],
    }


_REASON_MAX_CHARS = 200


def _sanitize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Truncate any ``reason`` fields to bound chain-of-thought leakage.

    The spec invariant says trajectories must NOT record hidden CoT. A naive
    LLM may dump its scratchpad into the ``reason`` field of a patch op or
    the top-level finalize reason. We keep a short prefix as documentation
    and discard the rest.
    """
    out = dict(action)
    if isinstance(out.get("reason"), str) and len(out["reason"]) > _REASON_MAX_CHARS:
        out["reason"] = out["reason"][:_REASON_MAX_CHARS] + "...[truncated]"
    if "patch_ops" in out and isinstance(out["patch_ops"], list):
        out["patch_ops"] = [_sanitize_op(op) for op in out["patch_ops"]]
    return out


def _sanitize_op(op: Any) -> Any:
    if not isinstance(op, dict):
        return op
    op = dict(op)
    if isinstance(op.get("reason"), str) and len(op["reason"]) > _REASON_MAX_CHARS:
        op["reason"] = op["reason"][:_REASON_MAX_CHARS] + "...[truncated]"
    return op


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"
