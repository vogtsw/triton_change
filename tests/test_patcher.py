"""Unit tests for triton_change.patcher."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_change.patcher import (
    PatchError,
    PatchResult,
    apply_patch_ops,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "task_000001"


# ---------- Helpers ----------


SIMPLE_SRC = '''\
import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
LN_EPS = 1e-5


@triton.jit
def gelu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = x * x
    tl.store(Y_ptr + offsets, y, mask=mask)


def model_forward(x):
    n = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    gelu_kernel[grid](x, y, n, BLOCK_SIZE=1024, num_warps=4)
    return y
'''


# ---------- update_constant ----------


def test_update_constant_basic(tmp_path):
    ops = [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "HIDDEN_SIZE",
        "old_value": 768,
        "new_value": 1024,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded, res.error
    assert "HIDDEN_SIZE = 1024" in res.candidate_text
    assert "HIDDEN_SIZE = 768" not in res.candidate_text


def test_update_constant_old_value_mismatch(tmp_path):
    ops = [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "HIDDEN_SIZE",
        "old_value": 999,
        "new_value": 1024,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert not res.all_succeeded
    assert "old_value mismatch" in res.op_results[0].detail


def test_update_constant_missing_name(tmp_path):
    ops = [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "DOES_NOT_EXIST",
        "new_value": 1,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert not res.all_succeeded
    assert "not found" in res.op_results[0].detail


def test_update_constant_float(tmp_path):
    ops = [{
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": "LN_EPS",
        "old_value": 1e-5,
        "new_value": 1e-6,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "LN_EPS = 1e-06" in res.candidate_text


def test_update_constant_two_ops(tmp_path):
    ops = [
        {
            "operation": "update_constant",
            "path": "candidate_model_triton.py",
            "constant_name": "HIDDEN_SIZE",
            "old_value": 768,
            "new_value": 1024,
        },
        {
            "operation": "update_constant",
            "path": "candidate_model_triton.py",
            "constant_name": "INTERMEDIATE_SIZE",
            "old_value": 3072,
            "new_value": 4096,
        },
    ]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "HIDDEN_SIZE = 1024" in res.candidate_text
    assert "INTERMEDIATE_SIZE = 4096" in res.candidate_text


# ---------- update_kernel_meta ----------


def test_update_kernel_meta_block_size(tmp_path):
    ops = [{
        "operation": "update_kernel_meta",
        "path": "candidate_model_triton.py",
        "kernel_name": "gelu_kernel",
        "meta_name": "BLOCK_SIZE",
        "new_value": 2048,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "BLOCK_SIZE=2048" in res.candidate_text


def test_update_kernel_meta_num_warps(tmp_path):
    ops = [{
        "operation": "update_kernel_meta",
        "path": "candidate_model_triton.py",
        "kernel_name": "gelu_kernel",
        "meta_name": "num_warps",
        "new_value": 8,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "num_warps=8" in res.candidate_text


# ---------- replace_function / replace_kernel_body ----------


def test_replace_function(tmp_path):
    new_code = '''def model_forward(x):
    return x * 2.0
'''
    ops = [{
        "operation": "replace_function",
        "path": "candidate_model_triton.py",
        "function_name": "model_forward",
        "new_code": new_code,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "return x * 2.0" in res.candidate_text


def test_replace_kernel_body(tmp_path):
    new_body = '''pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < N
x = tl.load(X_ptr + offsets, mask=mask)
y = x + 1.0
tl.store(Y_ptr + offsets, y, mask=mask)'''
    ops = [{
        "operation": "replace_kernel_body",
        "path": "candidate_model_triton.py",
        "kernel_name": "gelu_kernel",
        "new_body": new_body,
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "y = x + 1.0" in res.candidate_text
    # Decorator and signature still present
    assert "@triton.jit" in res.candidate_text
    assert "def gelu_kernel" in res.candidate_text


# ---------- regex_replace / full_file_replace ----------


def test_regex_replace_no_match_fails(tmp_path):
    ops = [{
        "operation": "regex_replace",
        "path": "candidate_model_triton.py",
        "pattern": "ZZZ_NEVER_MATCHES",
        "replacement": "x",
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert not res.all_succeeded


def test_regex_replace_works(tmp_path):
    ops = [{
        "operation": "regex_replace",
        "path": "candidate_model_triton.py",
        "pattern": r"HIDDEN_SIZE\s*=\s*768",
        "replacement": "HIDDEN_SIZE = 999",
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert "HIDDEN_SIZE = 999" in res.candidate_text


def test_full_file_replace(tmp_path):
    ops = [{
        "operation": "full_file_replace",
        "path": "candidate_model_triton.py",
        "new_code": "x = 1\n",
    }]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert res.all_succeeded
    assert res.candidate_text == "x = 1\n"


# ---------- Path safety ----------


def test_path_safety_dotdot(tmp_path):
    ops = [{
        "operation": "update_constant",
        "path": "../escape.py",
        "constant_name": "X",
        "new_value": 1,
    }]
    res = apply_patch_ops("X = 0\n", ops, workspace=tmp_path)
    assert not res.all_succeeded
    assert ".." in res.op_results[0].detail or "escape" in res.op_results[0].detail


def test_path_safety_absolute(tmp_path):
    ops = [{
        "operation": "update_constant",
        "path": "/etc/x.py",
        "constant_name": "X",
        "new_value": 1,
    }]
    res = apply_patch_ops("X = 0\n", ops, workspace=tmp_path)
    assert not res.all_succeeded


# ---------- Atomicity: failure leaves no candidate file ----------


def test_atomic_no_partial_write(tmp_path):
    ops = [
        {
            "operation": "update_constant",
            "path": "candidate_model_triton.py",
            "constant_name": "HIDDEN_SIZE",
            "old_value": 768,
            "new_value": 1024,
        },
        {
            "operation": "update_constant",
            "path": "candidate_model_triton.py",
            "constant_name": "DOES_NOT_EXIST",
            "new_value": 0,
        },
    ]
    res = apply_patch_ops(SIMPLE_SRC, ops, workspace=tmp_path)
    assert not res.all_succeeded
    assert not (tmp_path / "candidate_model_triton.py").exists()


# ---------- Integration: oracles for all generated tasks ----------


@pytest.mark.parametrize("task_id", [f"task_{i:06d}" for i in range(1, 21)])
def test_surgical_oracle_round_trip(task_id, tmp_path):
    """For every generated task: applying oracle/patch_ops.json to old_model_triton.py
    must produce a file whose AST matches oracle/new_model_triton.py (modulo docstring)."""
    task_dir = REPO_ROOT / "tasks" / task_id
    if not task_dir.exists():
        pytest.skip(f"{task_id} not generated yet")

    src = (task_dir / "old_model_triton.py").read_text(encoding="utf-8")
    ops = json.loads((task_dir / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]

    res = apply_patch_ops(src, ops, workspace=tmp_path)
    assert res.all_succeeded, res.error

    import ast

    def _strip_docstring(tree: ast.Module) -> ast.Module:
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body = tree.body[1:]
        return tree

    cand_tree = _strip_docstring(ast.parse(res.candidate_text))
    oracle = (task_dir / "oracle" / "new_model_triton.py").read_text(encoding="utf-8")
    oracle_tree = _strip_docstring(ast.parse(oracle))
    assert ast.dump(cand_tree, annotate_fields=False) == ast.dump(oracle_tree, annotate_fields=False)


@pytest.mark.parametrize("task_id", [f"task_{i:06d}" for i in (25, 50, 75, 100)])
def test_phase3b_sample_oracle_round_trip(task_id, tmp_path):
    """Spot-check Phase 3b generated tasks (21-100)."""
    task_dir = REPO_ROOT / "tasks" / task_id
    if not task_dir.exists():
        pytest.skip(f"{task_id} not generated yet")

    src = (task_dir / "old_model_triton.py").read_text(encoding="utf-8")
    ops = json.loads((task_dir / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]

    res = apply_patch_ops(src, ops, workspace=tmp_path)
    assert res.all_succeeded, res.error

    import ast

    def _strip_docstring(tree: ast.Module) -> ast.Module:
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body = tree.body[1:]
        return tree

    cand_tree = _strip_docstring(ast.parse(res.candidate_text))
    oracle = (task_dir / "oracle" / "new_model_triton.py").read_text(encoding="utf-8")
    oracle_tree = _strip_docstring(ast.parse(oracle))
    assert ast.dump(cand_tree, annotate_fields=False) == ast.dump(oracle_tree, annotate_fields=False)


def test_oracle_patch_produces_oracle_file(tmp_path):
    """Apply oracle/patch_ops.json to old_model_triton.py and verify the resulting
    candidate matches oracle/new_model_triton.py modulo the module docstring.

    Constants and code bodies must match the oracle exactly. Docstrings differ
    intentionally (old vs oracle) and are NOT part of what update_constant patches.
    """
    src = (TASK_DIR / "old_model_triton.py").read_text(encoding="utf-8")
    ops = json.loads((TASK_DIR / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]

    res = apply_patch_ops(src, ops, workspace=tmp_path)
    assert res.all_succeeded, res.error
    assert "HIDDEN_SIZE = 1024" in res.candidate_text
    assert "INTERMEDIATE_SIZE = 4096" in res.candidate_text
    assert "HIDDEN_SIZE = 768" not in res.candidate_text
    assert "INTERMEDIATE_SIZE = 3072" not in res.candidate_text

    import ast
    oracle = (TASK_DIR / "oracle" / "new_model_triton.py").read_text(encoding="utf-8")
    cand_tree = ast.parse(res.candidate_text)
    oracle_tree = ast.parse(oracle)
    # Drop the module docstring (first Expr(Constant) if present) before comparing
    def _strip_docstring(tree: ast.Module) -> ast.Module:
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body = tree.body[1:]
        return tree
    cand_dump = ast.dump(_strip_docstring(cand_tree), annotate_fields=False)
    oracle_dump = ast.dump(_strip_docstring(oracle_tree), annotate_fields=False)
    assert cand_dump == oracle_dump, "candidate AST (sans docstring) does not match oracle"
