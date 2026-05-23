"""Validate that task_000001 conforms to the v2 schemas.

These tests do NOT require torch / triton / GPU. They only check JSON schemas
and the presence of required source files.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = REPO_ROOT / "schemas"
TASK_DIR = REPO_ROOT / "tasks" / "task_000001"


def _load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------- task_000001 conformance ----------


def test_meta_validates_against_schema():
    schema = _load(SCHEMAS / "task_schema.json")
    instance = _load(TASK_DIR / "meta.json")
    jsonschema.validate(instance=instance, schema=schema)


def test_oracle_patch_ops_validates_against_schema():
    schema = _load(SCHEMAS / "patch_ops_schema.json")
    instance = _load(TASK_DIR / "oracle" / "patch_ops.json")
    jsonschema.validate(instance=instance, schema=schema)


def test_required_files_present():
    must_exist = [
        TASK_DIR / "old_model_triton.py",
        TASK_DIR / "oracle" / "new_model_triton.py",
        TASK_DIR / "oracle" / "diff_summary.json",
        TASK_DIR / "hidden_eval" / "input_specs.json",
        TASK_DIR / "hidden_eval" / "semantic_change_labels.json",
        TASK_DIR / "hidden_eval" / "reference_forward.py",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in must_exist if not p.exists()]
    assert not missing, f"missing files: {missing}"


def test_old_triton_has_expected_constants():
    src = (TASK_DIR / "old_model_triton.py").read_text(encoding="utf-8")
    assert "HIDDEN_SIZE = 768" in src
    assert "INTERMEDIATE_SIZE = 3072" in src
    assert "model_forward" in src


def test_new_triton_has_target_constants():
    src = (TASK_DIR / "oracle" / "new_model_triton.py").read_text(encoding="utf-8")
    assert "HIDDEN_SIZE = 1024" in src
    assert "INTERMEDIATE_SIZE = 4096" in src
    assert "model_forward" in src


def test_oracle_patch_count_matches_meta():
    inst = _load(TASK_DIR / "oracle" / "patch_ops.json")
    meta = _load(TASK_DIR / "meta.json")
    assert len(inst["ops"]) == meta["expected_oracle_patch_ops"]


# ---------- schema strictness ----------


def test_schema_rejects_unknown_operation():
    schema = _load(SCHEMAS / "patch_ops_schema.json")
    bad = {
        "task_id": "task_test",
        "ops": [{"operation": "delete_everything", "path": "candidate_x.py"}],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_schema_rejects_dotdot_path():
    schema = _load(SCHEMAS / "patch_ops_schema.json")
    bad = {
        "task_id": "task_test",
        "ops": [
            {
                "operation": "update_constant",
                "path": "../candidate.py",
                "constant_name": "X",
                "new_value": 1,
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_schema_rejects_absolute_path():
    schema = _load(SCHEMAS / "patch_ops_schema.json")
    bad = {
        "task_id": "task_test",
        "ops": [
            {
                "operation": "update_constant",
                "path": "/etc/candidate.py",
                "constant_name": "X",
                "new_value": 1,
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_schema_rejects_update_constant_without_new_value():
    schema = _load(SCHEMAS / "patch_ops_schema.json")
    bad = {
        "task_id": "task_test",
        "ops": [
            {
                "operation": "update_constant",
                "path": "candidate_model_triton.py",
                "constant_name": "HIDDEN_SIZE",
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_trajectory_schema_smoke():
    schema = _load(SCHEMAS / "trajectory_schema.json")
    sample = {
        "task_id": "task_000001",
        "tier": "MVP-easy",
        "agent_model": "deepseek-chat",
        "steps": [
            {
                "step_idx": 0,
                "observation": {"onnx_diff": {}, "code_summary": {}, "last_error": None},
                "action": {"tool": "apply_patch_ops", "patch_ops": []},
                "tool_result": {"static_check": "pass", "correctness": "fail"},
                "step_reward": -0.2,
                "reward_breakdown": {"syntax_pass": 0.1, "shape_match": -0.5, "small_patch": 0.1},
                "done": False,
                "failure_class": "shape_mismatch",
            }
        ],
        "success": False,
        "final_reward": -0.2,
        "total_steps": 1,
        "total_patch_ops": 1,
    }
    jsonschema.validate(instance=sample, schema=schema)


def test_trajectory_schema_rejects_unknown_tool():
    schema = _load(SCHEMAS / "trajectory_schema.json")
    bad = {
        "task_id": "task_000001",
        "steps": [
            {
                "step_idx": 0,
                "observation": {},
                "action": {"tool": "rm_rf"},
                "tool_result": {},
                "step_reward": 0.0,
                "done": True,
            }
        ],
        "success": False,
        "final_reward": 0.0,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)
