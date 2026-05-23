"""Tests for 100-task catalog (Phase 3b)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task_catalog import build_all_configs, tier_counts  # noqa: E402


def test_catalog_has_100_tasks():
    configs = build_all_configs()
    assert len(configs) == 100


def test_tier_ratio_5_3_2():
    counts = tier_counts()
    assert counts["MVP-easy"] == 50
    assert counts["MVP-medium"] == 30
    assert counts["MVP-hard"] == 20


def test_unique_task_ids():
    configs = build_all_configs()
    ids = [c.task_id for c in configs]
    assert len(ids) == len(set(ids))
    assert ids[0] == "task_000001"
    assert ids[-1] == "task_000100"
