"""Tests for triton_change.agent.tools."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from triton_change.agent.tools import (
    apply_patch_ops_tool,
    correctness_check_tool,
    inspect_code_region_tool,
    static_check_tool,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "task_000001"


def _stage_task(tmp_path: Path) -> Path:
    """Copy task_000001 into tmp_path so tests don't pollute the source tree."""
    dst = tmp_path / "task_000001"
    shutil.copytree(TASK_DIR, dst)
    # Drop any leftover candidate file
    cand = dst / "candidate_model_triton.py"
    if cand.exists():
        cand.unlink()
    sandbox = dst / "sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    return dst


def test_apply_patch_ops_tool_succeeds_and_writes_candidate(tmp_path):
    td = _stage_task(tmp_path)
    ops = json.loads((td / "oracle" / "patch_ops.json").read_text())["ops"]
    res = apply_patch_ops_tool(td, ops)
    assert res.success
    assert res.tool == "apply_patch_ops"
    assert (td / "candidate_model_triton.py").exists()


def test_apply_patch_ops_tool_failure_returns_failure_class(tmp_path):
    td = _stage_task(tmp_path)
    bad_ops = [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "DOES_NOT_EXIST",
        "new_value": 0,
    }]
    res = apply_patch_ops_tool(td, bad_ops)
    assert not res.success
    assert res.failure_class == "patch_apply_error"


def test_static_check_tool_pass_after_oracle_apply(tmp_path):
    td = _stage_task(tmp_path)
    ops = json.loads((td / "oracle" / "patch_ops.json").read_text())["ops"]
    apply_patch_ops_tool(td, ops)
    res = static_check_tool(td / "candidate_model_triton.py")
    assert res.success
    assert res.payload["passed"] is True


def test_static_check_tool_fails_on_syntax(tmp_path):
    td = _stage_task(tmp_path)
    bad = td / "candidate_model_triton.py"
    bad.write_text("def model_forward(:\n    pass\n", encoding="utf-8")
    res = static_check_tool(bad)
    assert not res.success
    assert res.failure_class == "syntax"


def test_static_check_tool_fails_on_unsafe_import(tmp_path):
    td = _stage_task(tmp_path)
    bad = td / "candidate_model_triton.py"
    bad.write_text("import os\n\ndef model_forward(x):\n    return x\n", encoding="utf-8")
    res = static_check_tool(bad)
    assert not res.success
    assert res.failure_class == "import"


def test_inspect_code_region_function(tmp_path):
    td = _stage_task(tmp_path)
    shutil.copy2(td / "old_model_triton.py", td / "candidate_model_triton.py")
    res = inspect_code_region_tool(td / "candidate_model_triton.py", "function:model_forward")
    assert res.success
    assert "def model_forward" in res.payload["content"]


def test_inspect_code_region_kernel(tmp_path):
    td = _stage_task(tmp_path)
    shutil.copy2(td / "old_model_triton.py", td / "candidate_model_triton.py")
    res = inspect_code_region_tool(td / "candidate_model_triton.py", "kernel:layernorm_fwd_kernel")
    assert res.success
    assert "@triton.jit" in res.payload["content"]
    assert "layernorm_fwd_kernel" in res.payload["content"]


def test_inspect_code_region_constants(tmp_path):
    td = _stage_task(tmp_path)
    shutil.copy2(td / "old_model_triton.py", td / "candidate_model_triton.py")
    res = inspect_code_region_tool(td / "candidate_model_triton.py", "constants")
    assert res.success
    assert "HIDDEN_SIZE" in res.payload["content"]


def test_inspect_unknown_region(tmp_path):
    td = _stage_task(tmp_path)
    shutil.copy2(td / "old_model_triton.py", td / "candidate_model_triton.py")
    res = inspect_code_region_tool(td / "candidate_model_triton.py", "function:nonsuch")
    assert not res.success


def _have_eval_data() -> bool:
    return (TASK_DIR / "hidden_eval" / "weights.pt").exists()


@pytest.mark.skipif(not _have_eval_data(), reason="task_000001 dynamic data missing")
def test_correctness_check_tool_passes_with_torch_only_candidate(tmp_path):
    td = _stage_task(tmp_path)
    cpu_demo = td / "oracle" / "cpu_demo_new_model_triton.py"
    shutil.copy2(cpu_demo, td / "candidate_model_triton.py")
    res = correctness_check_tool(td / "candidate_model_triton.py", td, timeout=60.0, device="cpu")
    assert res.success
    assert res.payload["passed"] is True
