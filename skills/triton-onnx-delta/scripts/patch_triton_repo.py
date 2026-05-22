from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _add_project_src() -> None:
    here = Path(__file__).resolve()
    project = here.parents[3]
    sys.path.insert(0, str(project / "src"))


def main() -> None:
    _add_project_src()
    from triton_change.triton.patcher import apply_incremental_patch

    parser = argparse.ArgumentParser(description="Skill wrapper for Triton incremental patching.")
    parser.add_argument("triton_model_dir", type=Path)
    parser.add_argument("base_onnx", type=Path)
    parser.add_argument("target_onnx", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/skill_patch"))
    parser.add_argument("--model-name")
    args = parser.parse_args()
    plan = apply_incremental_patch(args.triton_model_dir, args.base_onnx, args.target_onnx, args.out, args.model_name)
    from triton_change.onnx_delta.diff import compact_change_dict

    public = asdict(plan)
    public["delta"] = {
        "change_count": len(plan.delta.get("changes", [])),
        "changes": [compact_change_dict(change) for change in plan.delta.get("changes", [])],
    }
    print(json.dumps(public, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
