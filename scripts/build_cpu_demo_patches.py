"""Build `oracle/cpu_demo_patch_ops.json` for one or more tasks.

Each task's `oracle/cpu_demo_new_model_triton.py` (torch-only equivalent of
the real oracle) is wrapped into a single `full_file_replace` patch op so
that `run_phase1.py` can demonstrate end-to-end success on machines without
OpenAI Triton.

Usage:
    python scripts/build_cpu_demo_patches.py                       # all tasks
    python scripts/build_cpu_demo_patches.py task_000001 task_000003

The CPU demo patch will trigger the spec's `oversized_patch` penalty (it is
a full-file replace) but will pass syntax/static/correctness, demonstrating
the full reward pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"


def build_for_task(task_id: str) -> Path | None:
    task_dir = TASKS_DIR / task_id
    src = task_dir / "oracle" / "cpu_demo_new_model_triton.py"
    if not src.exists():
        print(f"[skip] {task_id}: no oracle/cpu_demo_new_model_triton.py")
        return None

    code = src.read_text(encoding="utf-8")
    doc = {
        "task_id": task_id,
        "ops": [
            {
                "operation": "full_file_replace",
                "path": "candidate_model_triton.py",
                "new_code": code,
                "reason": (
                    "CPU demo (no Triton): full-file replace with torch-only equivalent. "
                    "Triggers oversized_patch penalty; the real oracle is "
                    "oracle/patch_ops.json (2 update_constant ops)."
                ),
            }
        ],
        "notes": "Demo path for verifying the Phase 1 pipeline end-to-end on machines without Triton.",
    }
    out = task_dir / "oracle" / "cpu_demo_patch_ops.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out.relative_to(REPO_ROOT)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_ids", nargs="*", help="Task IDs (e.g. task_000001). Default: all.")
    args = ap.parse_args()

    if not args.task_ids:
        task_ids = sorted(p.name for p in TASKS_DIR.iterdir() if p.is_dir() and p.name.startswith("task_"))
    else:
        task_ids = args.task_ids

    if not task_ids:
        print("[err] no tasks found.")
        return 1

    built = 0
    for tid in task_ids:
        if build_for_task(tid) is not None:
            built += 1
    print(f"[done] built CPU demo patches for {built}/{len(task_ids)} task(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
