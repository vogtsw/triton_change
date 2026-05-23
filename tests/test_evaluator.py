"""Tests for unified evaluator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from triton_change.evaluator import evaluate_oracle


def test_evaluate_oracle_cpu_demo():
    td = REPO_ROOT / "tasks" / "task_000001"
    if not td.exists():
        pytest.skip("task not generated")
    ev = evaluate_oracle(td, use_cpu_demo=True, device="cpu")
    assert ev.oracle_ast_match
    assert ev.success
    assert ev.reward.total >= 1.0
    d = ev.to_dict()
    assert d["task_id"] == "task_000001"
    assert "reward" in d
