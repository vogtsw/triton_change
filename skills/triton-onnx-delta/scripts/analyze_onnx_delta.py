from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_project_src() -> None:
    here = Path(__file__).resolve()
    project = here.parents[3]
    sys.path.insert(0, str(project / "src"))


def main() -> None:
    _add_project_src()
    from triton_change.onnx_delta.diff import compact_report, diff_onnx, report_to_dict

    parser = argparse.ArgumentParser(description="Skill wrapper for ONNX architecture delta analysis.")
    parser.add_argument("base_onnx", type=Path)
    parser.add_argument("target_onnx", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    report = diff_onnx(args.base_onnx, args.target_onnx)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    text = json.dumps(report_to_dict(report) if args.full else compact_report(report), indent=2, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
