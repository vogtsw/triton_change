"""Tests for oracle patch judge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from triton_change.evaluator.patch_judge import judge_patch_for_task  # noqa: E402


@pytest.mark.parametrize("task_id", ["task_000001", "task_000020"])
def test_oracle_self_judge(task_id):
    td = REPO_ROOT / "tasks" / task_id
    if not td.exists():
        pytest.skip(f"{task_id} not generated")
    ops = json.loads((td / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]
    r = judge_patch_for_task(td, ops)
    assert r.apply_ok
    assert r.ast_match_oracle
    assert r.correct


def test_wrong_constant_fails():
    td = REPO_ROOT / "tasks" / "task_000001"
    if not td.exists():
        pytest.skip("task_000001 not generated")
    ops = json.loads((td / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]
    bad = [dict(op) for op in ops]
    bad[0]["new_value"] = 999
    r = judge_patch_for_task(td, bad)
    assert r.apply_ok
    assert not r.ast_match_oracle
    assert not r.correct
