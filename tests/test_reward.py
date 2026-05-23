"""Unit tests for triton_change.reward."""
from __future__ import annotations

import pytest

from triton_change.reward import compute_reward


def _static_pass():
    return {
        "syntax_ok": True,
        "imports_ok": True,
        "danger_findings": [],
    }


def _correct_pass():
    return {
        "executed": True,
        "passed": True,
        "shape_match": True,
        "dtype_match": True,
        "failure_class": None,
        "max_abs_error": 1e-6,
        "max_rel_error": 1e-6,
    }


def test_perfect_run_with_2_constant_ops_and_label():
    """task_000001 oracle scenario.

    Components hit:
      syntax(0.10) + import(0.10) + callable(0.20) + shape(0.30) + dtype(0.20)
      + numerical(1.00) + small(0.10) + semantic(0.20) + bonus(0.50) = 2.70
    """
    ops = [
        {"operation": "update_constant", "constant_name": "HIDDEN_SIZE", "new_value": 1024},
        {"operation": "update_constant", "constant_name": "INTERMEDIATE_SIZE", "new_value": 4096},
    ]
    labels = [
        {
            "label": "shape_param_change",
            "affected_region_hint": "constant:HIDDEN_SIZE,constant:INTERMEDIATE_SIZE",
        }
    ]
    rb = compute_reward(_static_pass(), _correct_pass(), ops, labels)
    assert rb.success
    assert rb.failure_class is None
    assert rb.total == pytest.approx(2.70, abs=1e-6)
    expected = {
        "syntax_pass", "import_pass", "model_forward_callable",
        "output_shape_match", "output_dtype_match",
        "numerical_correctness_pass",
        "small_localized_patch", "semantic_label_addressed",
        "all_or_nothing_bonus",
    }
    assert set(rb.components.keys()) == expected


def test_syntax_error_short_circuits():
    rb = compute_reward(
        {"syntax_ok": False, "imports_ok": False, "danger_findings": []},
        None,
        [],
        [],
    )
    assert rb.failure_class == "syntax"
    assert rb.total == pytest.approx(-0.40)
    assert "syntax_error" in rb.components


def test_unsafe_code_terminal():
    static = {"syntax_ok": True, "imports_ok": True, "danger_findings": ["call to os.system at line 5"]}
    rb = compute_reward(static, None, [], [])
    assert rb.failure_class == "unsafe_code"
    assert rb.total == pytest.approx(0.10 + 0.10 - 1.00)


def test_shape_mismatch_no_bonus():
    cr = {
        "executed": True,
        "passed": False,
        "shape_match": False,
        "dtype_match": False,
        "failure_class": "shape_mismatch",
    }
    rb = compute_reward(_static_pass(), cr, [], [])
    assert rb.failure_class == "shape_mismatch"
    assert "all_or_nothing_bonus" not in rb.components
    assert "numerical_correctness_pass" not in rb.components


def test_numerical_diverge():
    cr = {
        "executed": True,
        "passed": False,
        "shape_match": True,
        "dtype_match": True,
        "failure_class": "numerical_diverge",
        "max_abs_error": 0.5,
    }
    rb = compute_reward(_static_pass(), cr, [], [])
    assert rb.failure_class == "numerical_diverge"
    # No numerical_correctness_pass; got numerical_mismatch instead
    assert "numerical_correctness_pass" not in rb.components
    assert "numerical_mismatch" in rb.components
    assert "all_or_nothing_bonus" not in rb.components


def test_oversized_patch_penalty():
    ops = [{"operation": "full_file_replace", "new_code": "x = 1\n"}]
    rb = compute_reward(_static_pass(), _correct_pass(), ops, [])
    assert "oversized_patch" in rb.components
    # full_file_replace blocks small_localized_patch too
    assert "small_localized_patch" not in rb.components


def test_small_patch_gated_on_shape_match():
    cr = {
        "executed": True,
        "passed": False,
        "shape_match": False,
        "dtype_match": False,
        "failure_class": "shape_mismatch",
    }
    ops = [{"operation": "update_constant", "constant_name": "X", "new_value": 1}]
    rb = compute_reward(_static_pass(), cr, ops, [])
    # Even though the patch is small, shape_match is False so gate fails
    assert "small_localized_patch" not in rb.components


def test_timeout_terminal():
    rb = compute_reward(None, None, [], [], timed_out=True)
    assert rb.failure_class == "timeout"
    assert rb.total == pytest.approx(-1.00)


def test_patch_apply_error():
    rb = compute_reward(None, None, [], [], patch_apply_error=True)
    assert rb.failure_class == "patch_apply_error"
    assert rb.total == pytest.approx(-0.30)


def test_repeated_same_error_penalty():
    cr = {
        "executed": True,
        "passed": False,
        "shape_match": True,
        "dtype_match": True,
        "failure_class": "numerical_diverge",
    }
    rb = compute_reward(_static_pass(), cr, [], [], repeated_same_error=True)
    assert "repeated_same_error" in rb.components
    assert rb.components["repeated_same_error"] == pytest.approx(-0.20)


def test_zero_patch_passing_run():
    """Edge case: target == base, no patch needed, candidate passes."""
    rb = compute_reward(_static_pass(), _correct_pass(), [], [])
    # No semantic_label_addressed (no labels); no oversized; no repeated.
    # 2 constant ops → small (gate met). zero ops also counts as small.
    assert rb.success
    assert "small_localized_patch" in rb.components
    assert "all_or_nothing_bonus" in rb.components
