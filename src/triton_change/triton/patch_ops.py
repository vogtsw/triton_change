from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import onnx


@dataclass
class PatchOpResult:
    op: dict[str, Any]
    path: str
    action: str
    changed: bool
    detail: str


def inspect_triton_model(model_dir: Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir).as_posix()
        if path.suffix.lower() in {".onnx", ".data"}:
            files[rel] = {"kind": "binary", "bytes": path.stat().st_size}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        files[rel] = {
            "kind": "text",
            "bytes": path.stat().st_size,
            "line_count": len(text.splitlines()),
            "preview": "\n".join(text.splitlines()[:80]),
        }
    return {"model_dir": str(model_dir), "files": files}


def apply_patch_ops(
    source_model_dir: Path,
    out_dir: Path,
    model_name: str,
    ops: list[dict[str, Any]],
    target_onnx: Path,
    delta_report: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    source_model_dir = Path(source_model_dir)
    patched_model_dir = Path(out_dir) / model_name
    shutil.copytree(source_model_dir, patched_model_dir, dirs_exist_ok=True)
    results: list[dict[str, Any]] = []
    for op in ops:
        result = _apply_one(patched_model_dir, op, Path(target_onnx), delta_report or {})
        results.append(asdict(result))
    return patched_model_dir, results


def _apply_one(model_dir: Path, op: dict[str, Any], target_onnx: Path, delta_report: dict[str, Any]) -> PatchOpResult:
    operation = op.get("operation")
    rel_path = op.get("path", "")
    path = _safe_path(model_dir, rel_path)

    if operation == "copy_target_onnx":
        shutil.copy2(target_onnx, path)
        external_files = _copy_external_data_files(target_onnx, path.parent)
        detail = f"Copied {target_onnx} to {rel_path}."
        if external_files:
            detail += " Copied external data: " + ", ".join(external_files) + "."
        return PatchOpResult(op, rel_path, operation, True, detail)

    if operation == "regex_replace":
        text = path.read_text(encoding="utf-8")
        pattern = op["pattern"]
        replacement = op["replacement"]
        new_text, count = re.subn(pattern, replacement, text, count=op.get("count", 0), flags=re.MULTILINE | re.DOTALL)
        if count == 0:
            raise ValueError(f"regex_replace matched nothing in {rel_path}: {pattern}")
        path.write_text(new_text, encoding="utf-8")
        return PatchOpResult(op, rel_path, operation, new_text != text, f"Replaced {count} match(es).")

    if operation == "replace_text":
        text = path.read_text(encoding="utf-8")
        old = op["old"]
        new = op["new"]
        if old not in text:
            raise ValueError(f"replace_text old snippet not found in {rel_path}")
        path.write_text(text.replace(old, new, op.get("count", 1)), encoding="utf-8")
        return PatchOpResult(op, rel_path, operation, old != new, "Applied literal replacement.")

    if operation == "write_json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(op.get("content", {}), indent=2, ensure_ascii=False), encoding="utf-8")
        return PatchOpResult(op, rel_path, operation, True, "Wrote JSON file.")

    if operation == "write_delta_report":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(delta_report, indent=2, ensure_ascii=False), encoding="utf-8")
        return PatchOpResult(op, rel_path, operation, True, "Wrote full ONNX delta report from tool context.")

    raise ValueError(f"Unsupported patch operation: {operation}")


def write_patch_summary(
    patched_model_dir: Path,
    ops: list[dict[str, Any]],
    results: list[dict[str, Any]],
    delta_changes: list[dict[str, Any]],
    token_usage: list[dict[str, Any]],
) -> Path:
    path = Path(patched_model_dir) / "PATCH_SUMMARY.md"
    lines = ["# Patch Summary", ""]
    lines.append("## Files Changed")
    for result in results:
        lines.append(f"- `{result['path']}`: {result['action']} - {result['detail']}")
    lines.extend(["", "## ONNX Differences Used"])
    for change in delta_changes:
        lines.append(f"- `{change.get('category')}:{change.get('key')}` - {change.get('impact')}")
    lines.extend(["", "## Token Usage"])
    if token_usage:
        for item in token_usage:
            lines.append(
                f"- `{item.get('name')}`: input={item.get('input_tokens', 0)}, output={item.get('output_tokens', 0)}, provider={item.get('provider', 'unknown')}"
            )
    else:
        lines.append("- No LLM token usage recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _safe_path(root: Path, rel_path: str) -> Path:
    if not rel_path:
        raise ValueError("Patch operation path is required")
    root = root.resolve()
    path = (root / rel_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Patch path escapes model directory: {rel_path}")
    return path


def _copy_external_data_files(source_onnx: Path, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    try:
        model = onnx.load_model(source_onnx, load_external_data=False)
    except Exception:
        model = None

    locations: set[str] = set()
    if model is not None:
        for initializer in model.graph.initializer:
            for entry in initializer.external_data:
                if entry.key == "location" and entry.value:
                    locations.add(entry.value)

    sidecar = Path(str(source_onnx) + ".data")
    if sidecar.exists():
        locations.add(sidecar.name)

    for location in sorted(locations):
        src = (source_onnx.parent / location).resolve()
        if not src.exists() or not src.is_file():
            continue
        dest = dest_dir / Path(location).name
        shutil.copy2(src, dest)
        copied.append(dest.name)
    return copied
