from __future__ import annotations

import argparse
import json
from pathlib import Path

from triton_change.onnx_delta.diff import compact_report, diff_onnx, report_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze architecture differences between two ONNX graphs.")
    parser.add_argument("base_onnx", type=Path)
    parser.add_argument("target_onnx", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--full", action="store_true", help="Print the full graph summary instead of the compact report.")
    args = parser.parse_args()

    report = diff_onnx(args.base_onnx, args.target_onnx)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    text = json.dumps(report_to_dict(report) if args.full else compact_report(report), indent=2, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
