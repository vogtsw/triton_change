"""Unified tool wrapper around patcher / static_check / correctness.

Each tool returns a `ToolResult` with a stable shape so the runner can record
it in the trajectory and the policy can react to it. Tools never raise on
controlled failures — they return success=False with a populated
failure_class instead. They DO let unexpected exceptions bubble up so the
runner can surface them.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from triton_change.correctness import correctness_check as _correctness_check
from triton_change.patcher import apply_patch_ops as _apply_patch_ops
from triton_change.static_check import static_check as _static_check


__all__ = ["ToolResult", "apply_patch_ops_tool", "static_check_tool",
           "correctness_check_tool", "inspect_code_region_tool", "finalize_tool"]


VALID_TOOLS = {
    "apply_patch_ops",
    "inspect_code_region",
    "run_static_check",
    "run_correctness_check",
    "run_benchmark",  # stub for now
    "finalize",
}


@dataclass
class ToolResult:
    tool: str
    success: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- apply_patch_ops ----------


def apply_patch_ops_tool(
    task_dir: Path,
    ops: list[dict[str, Any]],
    source_path: Path | None = None,
    candidate_filename: str = "candidate_model_triton.py",
) -> ToolResult:
    """Apply ops to the current candidate (or to old_model_triton.py on first call).

    The candidate is written at `task_dir / candidate_filename`.
    """
    task_dir = Path(task_dir)
    candidate_path = task_dir / candidate_filename

    if source_path is None:
        # If a candidate already exists, patch on top of it; else start from old.
        source_path = candidate_path if candidate_path.exists() else (task_dir / "old_model_triton.py")
    source_text = Path(source_path).read_text(encoding="utf-8")

    res = _apply_patch_ops(source_text, ops, workspace=task_dir,
                           candidate_filename=candidate_filename)
    payload = res.to_dict()
    if not res.all_succeeded:
        return ToolResult(
            tool="apply_patch_ops",
            success=False,
            payload=payload,
            error=res.error,
            failure_class="patch_apply_error",
        )
    return ToolResult(
        tool="apply_patch_ops",
        success=True,
        payload=payload,
    )


# ---------- run_static_check ----------


def static_check_tool(candidate_path: Path) -> ToolResult:
    candidate_path = Path(candidate_path)
    if not candidate_path.exists():
        return ToolResult(
            tool="run_static_check",
            success=False,
            payload={},
            error=f"candidate file does not exist: {candidate_path}",
            failure_class="patch_apply_error",
        )
    r = _static_check(candidate_path)
    d = r.to_dict()
    if r.passed:
        return ToolResult(tool="run_static_check", success=True, payload=d)
    failure_class = (
        "syntax" if not r.syntax_ok else
        "unsafe_code" if r.danger_findings else
        "import" if not r.imports_ok else
        "syntax"  # fallback (e.g., no model_forward — treat as syntax-ish)
    )
    return ToolResult(
        tool="run_static_check",
        success=False,
        payload=d,
        error=(r.syntax_error or
               (f"bad imports: {r.bad_imports}" if r.bad_imports else None) or
               (f"danger: {r.danger_findings}" if r.danger_findings else None) or
               "model_forward missing or other static failure"),
        failure_class=failure_class,
    )


# ---------- run_correctness_check ----------


def correctness_check_tool(
    candidate_path: Path,
    task_dir: Path,
    timeout: float = 60.0,
    device: str = "cpu",
) -> ToolResult:
    candidate_path = Path(candidate_path)
    if not candidate_path.exists():
        return ToolResult(
            tool="run_correctness_check",
            success=False,
            payload={},
            error=f"candidate file does not exist: {candidate_path}",
            failure_class="patch_apply_error",
        )
    cr = _correctness_check(candidate_path, task_dir=task_dir, timeout=timeout, device=device)
    d = cr.to_dict()
    if cr.passed:
        return ToolResult(tool="run_correctness_check", success=True, payload=d)
    return ToolResult(
        tool="run_correctness_check",
        success=False,
        payload=d,
        error=cr.error,
        failure_class=cr.failure_class,
    )


# ---------- inspect_code_region ----------


_VALID_REGION_PREFIXES = ("function:", "kernel:")


def inspect_code_region_tool(candidate_path: Path, region: str) -> ToolResult:
    """Return the source of a named region.

    Region grammar (subset of patcher's):
        function:NAME    top-level function (incl. decorators)
        kernel:NAME      same as function (kernels are top-level functions)
        constants        all top-level Name = literal assignments
        imports          all import statements
        full             whole file
    """
    candidate_path = Path(candidate_path)
    if not candidate_path.exists():
        return ToolResult(
            tool="inspect_code_region",
            success=False,
            payload={},
            error=f"candidate file does not exist: {candidate_path}",
        )
    text = candidate_path.read_text(encoding="utf-8")

    snippet = _extract_region(text, region)
    if snippet is None:
        return ToolResult(
            tool="inspect_code_region",
            success=False,
            payload={"region": region},
            error=f"region {region!r} not found",
        )

    return ToolResult(
        tool="inspect_code_region",
        success=True,
        payload={"region": region, "content": snippet},
    )


def _extract_region(text: str, region: str) -> str | None:
    import ast as _ast
    if region == "full":
        return text
    if region == "imports":
        lines = text.splitlines(keepends=True)
        try:
            tree = _ast.parse(text)
        except SyntaxError:
            return None
        out: list[str] = []
        for node in tree.body:
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                start, end = node.lineno, node.end_lineno or node.lineno
                out.extend(lines[start - 1: end])
        return "".join(out) if out else ""
    if region == "constants":
        lines = text.splitlines(keepends=True)
        try:
            tree = _ast.parse(text)
        except SyntaxError:
            return None
        out = []
        for node in tree.body:
            if isinstance(node, _ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], _ast.Name):
                start, end = node.lineno, node.end_lineno or node.lineno
                out.extend(lines[start - 1: end])
        return "".join(out) if out else ""
    if region.startswith(_VALID_REGION_PREFIXES):
        name = region.split(":", 1)[1]
        try:
            tree = _ast.parse(text)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == name:
                start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                end = node.end_lineno or node.lineno
                lines = text.splitlines(keepends=True)
                return "".join(lines[start - 1: end])
        return None
    return None


# ---------- finalize ----------


def finalize_tool(reason: str) -> ToolResult:
    return ToolResult(
        tool="finalize",
        success=True,
        payload={"reason": reason},
    )
