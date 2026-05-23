"""Analyze ONNX pair and optionally refresh task oracle/diff_summary.

Usage:
    python scripts/analyze_onnx_diff.py tasks/task_000001
    python scripts/analyze_onnx_diff.py tasks/task_000001 --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from triton_change.diff_analyzer import analyze_onnx_pair, load_diff_from_task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--write", action="store_true", help="Overwrite oracle/diff_summary.json")
    args = ap.parse_args()

    td = args.task_dir.resolve()
    base, target = td / "base.onnx", td / "target.onnx"
    if base.exists() and target.exists():
        diff = analyze_onnx_pair(base, target)
        source = "onnx"
    else:
        diff = load_diff_from_task(td)
        source = "cached"

    print(json.dumps({"source": source, **diff}, indent=2, ensure_ascii=False))
    if args.write:
        out = td / "oracle" / "diff_summary.json"
        out.write_text(json.dumps(diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[wrote] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
