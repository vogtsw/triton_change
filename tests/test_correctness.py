"""Unit tests for triton_change.correctness.

These tests do NOT use Triton (which requires CUDA). Instead they use a
torch-only "shadow" candidate that mirrors task_000001's reference_forward.
This proves the sandbox + comparison + failure-class infrastructure works
on CPU on any platform.

Real Triton kernel verification still requires a Linux + GPU box; that is
exercised by `scripts/run_phase1.py --device cuda`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_change.correctness import correctness_check


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "task_000001"


def _have_generated_data() -> bool:
    return (
        (TASK_DIR / "hidden_eval" / "weights.pt").exists()
        and (TASK_DIR / "hidden_eval" / "test_inputs.pt").exists()
        and (TASK_DIR / "hidden_eval" / "target_outputs.pt").exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_generated_data(),
    reason="task_000001 dynamic data missing; run scripts/generate_task_000001.py first",
)


# A torch-only candidate equivalent to oracle/new_model_triton.py for HIDDEN=1024.
TORCH_ONLY_GOOD = '''\
import torch
import torch.nn.functional as F

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
LN_EPS = 1e-5


def model_forward(x, ln_w, ln_b, w1, b1, w2, b2):
    h = F.layer_norm(x, (HIDDEN_SIZE,), weight=ln_w, bias=ln_b, eps=LN_EPS)
    h = F.linear(h, w1, b1)
    h = F.gelu(h, approximate="tanh")
    h = F.linear(h, w2, b2)
    return h
'''


# Numerically wrong candidate (uses ReLU instead of GELU)
TORCH_ONLY_WRONG = '''\
import torch
import torch.nn.functional as F

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
LN_EPS = 1e-5


def model_forward(x, ln_w, ln_b, w1, b1, w2, b2):
    h = F.layer_norm(x, (HIDDEN_SIZE,), weight=ln_w, bias=ln_b, eps=LN_EPS)
    h = F.linear(h, w1, b1)
    h = F.relu(h)
    h = F.linear(h, w2, b2)
    return h
'''


# Wrong shape (missing the final linear)
TORCH_ONLY_BAD_SHAPE = '''\
import torch
import torch.nn.functional as F

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
LN_EPS = 1e-5


def model_forward(x, ln_w, ln_b, w1, b1, w2, b2):
    h = F.layer_norm(x, (HIDDEN_SIZE,), weight=ln_w, bias=ln_b, eps=LN_EPS)
    h = F.linear(h, w1, b1)
    return h  # shape [B, S, INTERMEDIATE] — wrong
'''


# Crash at runtime
TORCH_ONLY_RUNTIME_CRASH = '''\
import torch

def model_forward(x, ln_w, ln_b, w1, b1, w2, b2):
    raise RuntimeError("synthetic crash")
'''


# No model_forward
TORCH_ONLY_NO_FORWARD = '''\
import torch

def other(x):
    return x
'''


def _write_candidate(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "candidate_model_triton.py"
    p.write_text(content, encoding="utf-8")
    return p


def test_correctness_passes_on_good_candidate(tmp_path):
    cand = _write_candidate(tmp_path, TORCH_ONLY_GOOD)
    res = correctness_check(cand, TASK_DIR, timeout=60.0, device="cpu")
    assert res.executed
    assert res.passed, res.to_dict()
    assert res.shape_match
    assert res.dtype_match
    assert res.failure_class is None
    assert res.max_abs_error is not None and res.max_abs_error < 1e-4


def test_correctness_detects_numerical_diverge(tmp_path):
    cand = _write_candidate(tmp_path, TORCH_ONLY_WRONG)
    res = correctness_check(cand, TASK_DIR, timeout=60.0, device="cpu")
    assert res.executed
    assert not res.passed
    assert res.failure_class == "numerical_diverge"
    assert res.shape_match  # shape is still correct
    assert res.max_abs_error is not None and res.max_abs_error > 1e-2


def test_correctness_detects_shape_mismatch(tmp_path):
    cand = _write_candidate(tmp_path, TORCH_ONLY_BAD_SHAPE)
    res = correctness_check(cand, TASK_DIR, timeout=60.0, device="cpu")
    assert res.executed
    assert not res.passed
    assert res.failure_class == "shape_mismatch"


def test_correctness_detects_runtime_error(tmp_path):
    cand = _write_candidate(tmp_path, TORCH_ONLY_RUNTIME_CRASH)
    res = correctness_check(cand, TASK_DIR, timeout=60.0, device="cpu")
    assert res.executed
    assert not res.passed
    assert res.failure_class == "runtime"
    assert "synthetic crash" in (res.error or "")


def test_correctness_detects_missing_model_forward(tmp_path):
    cand = _write_candidate(tmp_path, TORCH_ONLY_NO_FORWARD)
    res = correctness_check(cand, TASK_DIR, timeout=60.0, device="cpu")
    assert res.executed
    assert not res.passed
    assert res.failure_class == "import"
    assert "model_forward" in (res.error or "")


def test_correctness_timeout(tmp_path):
    # Candidate that loops forever — verifies timeout handling
    src = '''\
def model_forward(x, ln_w, ln_b, w1, b1, w2, b2):
    while True:
        pass
'''
    cand = _write_candidate(tmp_path, src)
    res = correctness_check(cand, TASK_DIR, timeout=2.0, device="cpu")
    assert res.executed
    assert not res.passed
    assert res.failure_class == "timeout"
