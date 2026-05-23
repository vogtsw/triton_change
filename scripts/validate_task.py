"""Validate that a task directory has all required files conforming to v2 schemas.

Usage:
    python scripts/validate_task.py tasks/task_000001
    python scripts/validate_task.py tasks/task_000001 --strict   # fail on warnings too
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def validate(task_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not task_dir.is_dir():
        return [f"task dir not found: {task_dir}"], []

    required = [
        "meta.json",
        "old_model_triton.py",
        "oracle/new_model_triton.py",
        "oracle/patch_ops.json",
        "oracle/diff_summary.json",
        "hidden_eval/input_specs.json",
        "hidden_eval/semantic_change_labels.json",
        "hidden_eval/reference_forward.py",
    ]
    for rel in required:
        if not (task_dir / rel).exists():
            errors.append(f"missing required file: {rel}")

    recommended = [
        "base.onnx",
        "target.onnx",
        "hidden_eval/test_inputs.pt",
        "hidden_eval/target_outputs.pt",
        "hidden_eval/weights.pt",
    ]
    for rel in recommended:
        if not (task_dir / rel).exists():
            warnings.append(f"missing generated file: {rel} (run scripts/generate_task_*.py)")

    if (task_dir / "meta.json").exists():
        try:
            schema = load_json(SCHEMAS_DIR / "task_schema.json")
            jsonschema.validate(load_json(task_dir / "meta.json"), schema)
        except Exception as e:
            errors.append(f"meta.json schema error: {e}")

    if (task_dir / "oracle" / "patch_ops.json").exists():
        try:
            schema = load_json(SCHEMAS_DIR / "patch_ops_schema.json")
            jsonschema.validate(load_json(task_dir / "oracle" / "patch_ops.json"), schema)
        except Exception as e:
            errors.append(f"oracle/patch_ops.json schema error: {e}")

    return errors, warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_dir", type=Path)
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = p.parse_args()

    errors, warnings = validate(args.task_dir)

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    if args.strict and warnings:
        print("FAILED (strict): warnings present")
        sys.exit(1)

    print(f"OK: {args.task_dir} validates against schemas.")


if __name__ == "__main__":
    main()
