"""Tests for ONNX diff analyzer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from triton_change.diff_analyzer import analyze_onnx_pair, infer_semantic_labels


@pytest.fixture
def task001_dir():
    return REPO_ROOT / "tasks" / "task_000001"


def test_infer_semantic_labels_shape():
    raw = {"input_shape_changes": [{"name": "input", "old_shape": [1], "new_shape": [2]}]}
    labels = infer_semantic_labels(raw)
    assert any(l["label"] == "shape_param_change" for l in labels)


def test_analyze_onnx_pair_task001(task001_dir):
    base = task001_dir / "base.onnx"
    target = task001_dir / "target.onnx"
    if not base.exists() or not target.exists():
        pytest.skip("ONNX files not generated (run generate_tasks.py)")
    diff = analyze_onnx_pair(base, target)
    assert "raw_diff" in diff
    assert "semantic_labels" in diff
    assert diff["raw_diff"]["input_shape_changes"]
    blob = json.dumps(diff)
    assert len(blob) < 50_000


def test_diff_summary_matches_onnx(task001_dir):
    base = task001_dir / "base.onnx"
    if not base.exists():
        pytest.skip("ONNX not generated")
    cached = json.loads((task001_dir / "oracle" / "diff_summary.json").read_text(encoding="utf-8"))
    live = analyze_onnx_pair(task001_dir / "base.onnx", task001_dir / "target.onnx")
    assert cached["semantic_labels"][0]["label"] == live["semantic_labels"][0]["label"]
