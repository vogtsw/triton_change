"""End-to-end runner tests with deterministic policies (Mock / Oracle).

DeepSeek policy is exercised separately by `scripts/run_agent.py --policy deepseek`
because it requires a network call.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from triton_change.agent import AgentRunner, MockPolicy, OraclePolicy
from triton_change.agent.runner import VALID_AGENT_TOOLS
from triton_change.agent.trajectory import validate_trajectory


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_ID = "task_000001"
TASK_DIR = REPO_ROOT / "tasks" / TASK_ID


def _stage_task(tmp_path: Path) -> Path:
    dst = tmp_path / TASK_ID
    shutil.copytree(TASK_DIR, dst)
    for f in [dst / "candidate_model_triton.py", dst / "trajectory.json", dst / "reward.json"]:
        if f.exists():
            f.unlink()
    sb = dst / "sandbox"
    if sb.exists():
        shutil.rmtree(sb)
    return dst


def _have_eval_data() -> bool:
    return (TASK_DIR / "hidden_eval" / "weights.pt").exists()


pytestmark = pytest.mark.skipif(not _have_eval_data(),
                                reason="task_000001 dynamic data missing")


# ---------- Mock policy: success path ----------


def test_mock_policy_full_file_replace_path_completes(tmp_path):
    """The CPU-demo full_file_replace via mock should hit success in 2 steps."""
    td = _stage_task(tmp_path)
    cpu_demo = (td / "oracle" / "cpu_demo_new_model_triton.py").read_text(encoding="utf-8")
    actions = [
        {"tool": "apply_patch_ops", "patch_ops": [{
            "operation": "full_file_replace",
            "path": "candidate_model_triton.py",
            "new_code": cpu_demo,
        }]},
        {"tool": "finalize", "reason": "test"},
    ]
    runner = AgentRunner(td, MockPolicy(actions), max_steps=4, device="cpu",
                          correctness_timeout=60.0)
    result = runner.run()

    assert result.success is True
    assert result.final_reward > 0  # 2.10 with oversized penalty + bonus
    assert result.total_steps >= 1
    # Trajectory schema must validate
    validate_trajectory(result.to_dict())


# ---------- Mock policy: failure path with retry ----------


def test_mock_policy_first_fail_then_success(tmp_path):
    """Step 0 = bad patch (fails patch_apply); step 1 = correct full file replace."""
    td = _stage_task(tmp_path)
    cpu_demo = (td / "oracle" / "cpu_demo_new_model_triton.py").read_text(encoding="utf-8")
    actions = [
        {"tool": "apply_patch_ops", "patch_ops": [{
            "operation": "update_constant",
            "path": "candidate_model_triton.py",
            "constant_name": "NOT_A_REAL_NAME",
            "new_value": 0,
        }]},
        {"tool": "apply_patch_ops", "patch_ops": [{
            "operation": "full_file_replace",
            "path": "candidate_model_triton.py",
            "new_code": cpu_demo,
        }]},
        {"tool": "finalize", "reason": "test"},
    ]
    result = AgentRunner(td, MockPolicy(actions), max_steps=5, device="cpu",
                         correctness_timeout=60.0).run()

    assert result.total_steps >= 2
    # First step failed, second step succeeded
    assert result.steps[0].failure_class == "patch_apply_error"
    assert result.success is True


# ---------- Mock policy: max_patch_attempts cap ----------


def test_mock_policy_force_finalize_after_repeated_failures(tmp_path):
    """All patches are bad; runner should cap at max_patch_attempts."""
    td = _stage_task(tmp_path)
    bad_action = {"tool": "apply_patch_ops", "patch_ops": [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "NEVER_EXISTS",
        "new_value": 0,
    }]}
    actions = [bad_action] * 10
    result = AgentRunner(td, MockPolicy(actions), max_steps=10,
                         max_patch_attempts=3, max_repeated_errors=99,
                         device="cpu", correctness_timeout=60.0).run()

    assert result.success is False
    # We allow up to max_patch_attempts patch tries; cap should kick in
    assert result.total_steps <= 4  # 3 attempts + maybe 1 finalize step
    assert result.failure_class == "patch_apply_error"


# ---------- Mock policy: repeated-error breaker ----------


def test_repeated_same_error_terminates(tmp_path):
    td = _stage_task(tmp_path)
    bad_action = {"tool": "apply_patch_ops", "patch_ops": [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "NEVER",
        "new_value": 0,
    }]}
    actions = [bad_action] * 10
    result = AgentRunner(td, MockPolicy(actions), max_steps=10,
                         max_patch_attempts=99, max_repeated_errors=2,
                         device="cpu", correctness_timeout=60.0).run()
    assert result.total_steps <= 3  # repeated breaker fires after 2 same errors


# ---------- Oracle policy (Windows can verify pipeline; Triton import fails) ----------


def test_oracle_policy_runs_and_records_trajectory(tmp_path):
    td = _stage_task(tmp_path)
    runner = AgentRunner(td, OraclePolicy(td), max_steps=3, device="cpu",
                         correctness_timeout=60.0)
    result = runner.run()
    assert result.total_steps >= 1
    # On Windows without Triton, correctness will fail with "import"; on Linux+GPU it'll succeed.
    # In either case, trajectory schema must hold.
    validate_trajectory(result.to_dict())
    # First step must be apply_patch_ops
    assert result.steps[0].action["tool"] == "apply_patch_ops"


# ---------- Action coercion ----------


def test_runner_cleans_stale_candidate(tmp_path):
    """A leftover candidate from a previous run must not pollute step 0."""
    td = _stage_task(tmp_path)
    # Plant a stale candidate that has the OPPOSITE constants from old.
    (td / "candidate_model_triton.py").write_text(
        "HIDDEN_SIZE = 99999\nINTERMEDIATE_SIZE = 99999\nLN_EPS = 1e-5\n"
        "def model_forward(*args, **kwargs):\n    raise RuntimeError('stale')\n",
        encoding="utf-8",
    )
    actions = [
        {"tool": "inspect_code_region", "region": "constants"},
        {"tool": "finalize", "reason": "ok"},
    ]
    runner = AgentRunner(td, MockPolicy(actions), max_steps=2, device="cpu")
    result = runner.run()
    # After reset, step 0 inspect should NOT see HIDDEN_SIZE = 99999.
    inspect_step = result.steps[0]
    inspect_result = inspect_step.tool_result["results"][0]
    content = inspect_result["payload"].get("content", "")
    assert "99999" not in content
    assert "768" in content  # the original old_model value


def test_runner_truncates_long_reason_in_action(tmp_path):
    td = _stage_task(tmp_path)
    long_reason = "thinking " * 100  # > 200 chars
    actions = [
        {"tool": "apply_patch_ops", "patch_ops": [{
            "operation": "update_constant",
            "path": "candidate_model_triton.py",
            "constant_name": "HIDDEN_SIZE",
            "new_value": 1024,
            "reason": long_reason,
        }]},
        {"tool": "finalize", "reason": long_reason},
    ]
    runner = AgentRunner(td, MockPolicy(actions), max_steps=3, device="cpu")
    result = runner.run()
    # First step: reason of patch op should be truncated
    op0 = result.steps[0].action["patch_ops"][0]
    assert len(op0["reason"]) < 250
    assert "[truncated]" in op0["reason"]
    # Final step: top-level reason also truncated
    final_reason = result.steps[-1].action.get("reason", "")
    if len(long_reason) > 200:
        assert "[truncated]" in final_reason


def test_invalid_tool_is_coerced_to_finalize(tmp_path):
    td = _stage_task(tmp_path)
    actions = [{"tool": "totally_made_up_tool"}]
    result = AgentRunner(td, MockPolicy(actions), max_steps=2, device="cpu").run()
    assert result.steps[0].action["tool"] in VALID_AGENT_TOOLS
    assert result.steps[0].action["tool"] == "finalize"


# ---------- Trajectory shape ----------


def test_trajectory_has_required_top_level_fields(tmp_path):
    td = _stage_task(tmp_path)
    cpu_demo = (td / "oracle" / "cpu_demo_new_model_triton.py").read_text()
    actions = [{"tool": "apply_patch_ops", "patch_ops": [{
        "operation": "full_file_replace",
        "path": "candidate_model_triton.py",
        "new_code": cpu_demo,
    }]}, {"tool": "finalize", "reason": "test"}]
    traj = AgentRunner(td, MockPolicy(actions), max_steps=4, device="cpu",
                       correctness_timeout=60.0).run().to_dict()

    for k in ["task_id", "steps", "success", "final_reward",
              "total_steps", "total_patch_ops", "agent_model",
              "agent_run_id", "git_commit_sha"]:
        assert k in traj, f"missing {k}"
    for s in traj["steps"]:
        for k in ["step_idx", "observation", "action", "tool_result",
                  "step_reward", "reward_breakdown", "done"]:
            assert k in s, f"step missing {k}"
        assert s["action"]["tool"] in VALID_AGENT_TOOLS
