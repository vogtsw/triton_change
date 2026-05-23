"""Tests for SFT export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_sft import extract_sft_pairs  # noqa: E402


def test_extract_sft_pairs_from_minimal_trajectory():
    traj = {
        "task_id": "task_000001",
        "tier": "MVP-easy",
        "success": True,
        "steps": [
            {
                "step_idx": 0,
                "observation": {"code_summary": {"constants": []}, "hint": "try update_constant"},
                "action": {"tool": "apply_patch_ops", "patch_ops": [{"operation": "update_constant"}]},
                "step_reward": 0.5,
            },
            {
                "step_idx": 1,
                "observation": {},
                "action": {"tool": "finalize", "reason": "done"},
                "step_reward": 1.0,
            },
        ],
    }
    pairs = extract_sft_pairs(traj)
    assert len(pairs) == 1
    assert pairs[0]["action"]["tool"] == "apply_patch_ops"
    assert "prompt" in pairs[0]
    assert "response" in pairs[0]
