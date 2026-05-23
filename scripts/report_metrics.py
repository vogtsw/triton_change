"""Compute baseline metrics from trajectory JSONL (spec section 10).

Usage:
    python scripts/report_metrics.py trajectories/baseline_oracle.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compute_metrics(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trajectories) or 1
    successes = sum(1 for t in trajectories if t.get("success"))
    rewards = [float(t.get("final_reward", 0)) for t in trajectories]
    steps = [int(t.get("total_steps", 0)) for t in trajectories]
    patch_ops = [int(t.get("total_patch_ops", 0)) for t in trajectories]

    failure_classes: dict[str, int] = {}
    for t in trajectories:
        fc = t.get("failure_class") or "none"
        failure_classes[fc] = failure_classes.get(fc, 0) + 1

    oversized = sum(
        1 for t in trajectories
        if int(t.get("total_patch_ops", 0)) > 8
    )

    return {
        "count": len(trajectories),
        "end_to_end_success_rate": successes / n,
        "avg_final_reward": sum(rewards) / n,
        "avg_steps": sum(steps) / n,
        "avg_patch_ops": sum(patch_ops) / n,
        "oversized_patch_rate": oversized / n,
        "failure_class_distribution": failure_classes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    trajs = list(_iter_jsonl(args.input))
    metrics = compute_metrics(trajs)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
