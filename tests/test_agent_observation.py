"""Tests for triton_change.agent.observation."""
from __future__ import annotations

from pathlib import Path

from triton_change.agent.observation import (
    Observation,
    error_signature,
    extract_code_summary,
    failure_hint,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "task_000001"


def test_extract_code_summary_old_triton():
    cs = extract_code_summary(TASK_DIR / "old_model_triton.py")
    names = {c.name for c in cs.constants}
    assert {"HIDDEN_SIZE", "INTERMEDIATE_SIZE", "LN_EPS", "GELU_BLOCK_SIZE", "DTYPE_NAME"} <= names
    hidden = next(c for c in cs.constants if c.name == "HIDDEN_SIZE")
    assert hidden.value == 768
    kernel_names = {k.name for k in cs.kernels}
    assert kernel_names == {"layernorm_fwd_kernel", "gelu_act_kernel"}
    func_names = {f.name for f in cs.functions}
    assert {"_layernorm", "_gelu_tanh", "model_forward", "_compute_dtype"} <= func_names


def test_extract_code_summary_oracle():
    cs = extract_code_summary(TASK_DIR / "oracle" / "new_model_triton.py")
    hidden = next(c for c in cs.constants if c.name == "HIDDEN_SIZE")
    inter = next(c for c in cs.constants if c.name == "INTERMEDIATE_SIZE")
    assert hidden.value == 1024
    assert inter.value == 4096


def test_extract_code_summary_to_dict_serializable():
    import json
    cs = extract_code_summary(TASK_DIR / "old_model_triton.py")
    s = json.dumps(cs.to_dict())
    assert "HIDDEN_SIZE" in s


def test_failure_hint_known_classes():
    assert "shape" in failure_hint("shape_mismatch", repeated=False).lower()
    assert "epsilon" in failure_hint("numerical_diverge", repeated=False).lower()
    assert "import" in failure_hint("import", repeated=False).lower()


def test_failure_hint_repeated_adds_strategy_hint():
    h = failure_hint("shape_mismatch", repeated=True)
    assert h is not None
    assert "STRATEGY HINT" in h


def test_failure_hint_none_when_no_class():
    assert failure_hint(None) is None
    assert failure_hint("") is None


def test_error_signature_stable_for_same_input():
    a = error_signature({"failure_class": "shape_mismatch", "error": "shape: (1,128,768) vs (1,128,1024)"})
    b = error_signature({"failure_class": "shape_mismatch", "error": "shape: (1,128,768) vs (1,128,1024)"})
    assert a == b


def test_error_signature_changes_for_different_classes():
    a = error_signature({"failure_class": "shape_mismatch", "error": "x"})
    b = error_signature({"failure_class": "numerical_diverge", "error": "x"})
    assert a != b


def test_observation_to_dict_has_required_keys():
    obs = Observation(
        step_idx=0, task_id="task_000001",
        onnx_diff={"semantic_labels": []},
        code_summary={},
        remaining_steps=8,
    )
    d = obs.to_dict()
    for k in ["step_idx", "task_id", "onnx_diff", "code_summary",
              "last_action", "last_error", "hint", "remaining_steps",
              "repeated_same_error"]:
        assert k in d
