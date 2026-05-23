"""Patch applier for Triton single-file candidate code.

Implements the 8 patch op types from spec section 5.4:

    update_constant         Top-level `NAME = literal` assignment.
    update_kernel_meta      Kernel launch site keyword arg (BLOCK_SIZE, num_warps, ...).
    replace_function        Whole top-level function (incl. decorators).
    replace_kernel_body     Body of a `@triton.jit` function only.
    replace_region          Region addressed by name (`function:NAME` / `kernel:NAME`).
    insert_after_region     Insert code after a named region.
    regex_replace           Plain regex.sub (last-resort).
    full_file_replace       Replace whole file (heaviest hammer).

All edits are done by surgical text slicing using AST line/col info — we never
call `ast.unparse`, so comments and formatting are preserved.

Path safety: every op's `path` is resolved against a workspace dir; paths
containing `..`, absolute paths, or paths that resolve outside the workspace
are rejected at runtime (in addition to the schema-level checks).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


__all__ = [
    "apply_patch_ops",
    "PatchOpResult",
    "PatchResult",
    "PatchError",
]


class PatchError(Exception):
    """Raised when a patch op cannot be applied."""


@dataclass
class PatchOpResult:
    op: dict[str, Any]
    success: bool
    detail: str
    locator: str = ""


@dataclass
class PatchResult:
    candidate_path: Path | None
    candidate_text: str
    op_results: list[PatchOpResult]
    all_succeeded: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_path": str(self.candidate_path) if self.candidate_path else None,
            "all_succeeded": self.all_succeeded,
            "error": self.error,
            "op_results": [
                {"op": r.op, "success": r.success, "detail": r.detail, "locator": r.locator}
                for r in self.op_results
            ],
        }


# ---------- Public API ----------


def apply_patch_ops(
    source_text: str,
    ops: list[dict[str, Any]],
    workspace: Path,
    candidate_filename: str = "candidate_model_triton.py",
    write: bool = True,
) -> PatchResult:
    """Apply ops sequentially to source_text.

    On success: writes workspace/candidate_filename (if write=True) and returns
    a PatchResult with all_succeeded=True. On any op failure: stops, does NOT
    write, and returns a PatchResult whose op_results lists the failure.
    """
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    text = source_text
    results: list[PatchOpResult] = []

    for op in ops:
        op_type = op.get("operation")
        try:
            _check_path_safety(workspace, op.get("path", ""))
            handler = _OP_HANDLERS.get(op_type)
            if handler is None:
                raise PatchError(f"unknown operation: {op_type!r}")
            text, locator = handler(text, op)
            results.append(PatchOpResult(op=op, success=True, detail="ok", locator=locator))
        except Exception as e:
            results.append(PatchOpResult(op=op, success=False, detail=str(e)))
            return PatchResult(
                candidate_path=None,
                candidate_text=source_text,
                op_results=results,
                all_succeeded=False,
                error=f"op #{len(results)} ({op_type}) failed: {e}",
            )

    candidate_path = workspace / candidate_filename
    if write:
        candidate_path.write_text(text, encoding="utf-8")
    return PatchResult(
        candidate_path=candidate_path,
        candidate_text=text,
        op_results=results,
        all_succeeded=True,
    )


# ---------- Path safety ----------


def _check_path_safety(workspace: Path, rel: str) -> Path:
    if not rel:
        raise PatchError("patch op path is empty")
    p = Path(rel)
    if p.is_absolute() or rel.startswith(("/", "\\")):
        raise PatchError(f"absolute path forbidden: {rel}")
    if ".." in p.parts:
        raise PatchError(f"'..' segment forbidden in path: {rel}")
    full = (workspace / rel).resolve()
    try:
        full.relative_to(workspace)
    except ValueError:
        raise PatchError(f"path escapes workspace: {rel}") from None
    return full


# ---------- Slice helpers ----------


def _line_col_to_offset(text: str, line: int, col: int) -> int:
    """Convert (1-indexed line, 0-indexed col) to character offset."""
    cur = 0
    for i, ln in enumerate(text.splitlines(keepends=True), start=1):
        if i == line:
            return cur + col
        cur += len(ln)
    return cur + col


def _slice_replace(text: str, start_line: int, start_col: int, end_line: int, end_col: int, replacement: str) -> str:
    so = _line_col_to_offset(text, start_line, start_col)
    eo = _line_col_to_offset(text, end_line, end_col)
    return text[:so] + replacement + text[eo:]


# ---------- Op handlers ----------


def _op_update_constant(text: str, op: dict[str, Any]) -> tuple[str, str]:
    name = op["constant_name"]
    new_value = op["new_value"]
    old_value_expected = op.get("old_value")

    tree = ast.parse(text)
    matches: list[ast.Assign] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            matches.append(node)

    if not matches:
        raise PatchError(f"top-level constant {name!r} not found")
    if len(matches) > 1:
        raise PatchError(f"multiple top-level assignments to {name!r}; ambiguous")

    value = matches[0].value

    if old_value_expected is not None:
        try:
            current = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            current = "<non-literal>"
        if current != old_value_expected:
            raise PatchError(
                f"old_value mismatch for {name}: expected {old_value_expected!r}, got {current!r}"
            )

    new_text = _slice_replace(
        text, value.lineno, value.col_offset, value.end_lineno, value.end_col_offset,
        repr(new_value),
    )
    return new_text, f"line {value.lineno} cols {value.col_offset}-{value.end_col_offset}"


def _op_update_kernel_meta(text: str, op: dict[str, Any]) -> tuple[str, str]:
    kernel_name = op["kernel_name"]
    meta_name = op["meta_name"]
    new_value = op["new_value"]

    tree = ast.parse(text)
    matches: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == kernel_name
        ):
            matches.append(node)

    if not matches:
        raise PatchError(f"no launch site for kernel {kernel_name!r} found")

    matches.sort(key=lambda n: (n.lineno, n.col_offset), reverse=True)

    locators = []
    for call in matches:
        kw = next((k for k in call.keywords if k.arg == meta_name), None)
        if kw is None:
            raise PatchError(
                f"keyword {meta_name!r} not present in {kernel_name} launch at line {call.lineno}"
            )
        v = kw.value
        text = _slice_replace(
            text, v.lineno, v.col_offset, v.end_lineno, v.end_col_offset,
            repr(new_value),
        )
        locators.append(f"line {v.lineno}")
    return text, "; ".join(locators)


def _find_top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    if not matches:
        raise PatchError(f"function {name!r} not found at module level")
    if len(matches) > 1:
        raise PatchError(f"multiple definitions of {name!r}")
    return matches[0]


def _function_full_span(fn: ast.FunctionDef) -> tuple[int, int]:
    start_line = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
    end_line = fn.end_lineno or fn.lineno
    return start_line, end_line


def _op_replace_function(text: str, op: dict[str, Any]) -> tuple[str, str]:
    name = op["function_name"]
    new_code = op["new_code"].rstrip("\n") + "\n"

    tree = ast.parse(text)
    fn = _find_top_level_function(tree, name)
    start_line, end_line = _function_full_span(fn)

    lines = text.splitlines(keepends=True)
    new_text = "".join(lines[: start_line - 1]) + new_code + "".join(lines[end_line:])
    return new_text, f"lines {start_line}-{end_line}"


def _op_replace_kernel_body(text: str, op: dict[str, Any]) -> tuple[str, str]:
    name = op["kernel_name"]
    new_body = op["new_body"]

    tree = ast.parse(text)
    fn = _find_top_level_function(tree, name)
    if not fn.body:
        raise PatchError(f"kernel {name!r} has empty body")

    body_start = fn.body[0].lineno
    body_end = fn.end_lineno or fn.body[-1].end_lineno or body_start

    lines = text.splitlines(keepends=True)
    def_line = lines[fn.lineno - 1]
    def_indent = len(def_line) - len(def_line.lstrip(" "))
    body_indent = " " * (def_indent + 4)

    new_body_lines = new_body.rstrip("\n").split("\n")
    indented = "\n".join(
        (body_indent + ln if ln.strip() else "") for ln in new_body_lines
    ) + "\n"

    new_text = "".join(lines[: body_start - 1]) + indented + "".join(lines[body_end:])
    return new_text, f"lines {body_start}-{body_end}"


def _op_replace_region(text: str, op: dict[str, Any]) -> tuple[str, str]:
    region = op["region"]
    new_code = op["new_code"]
    if region.startswith("function:"):
        return _op_replace_function(
            text, {"function_name": region[len("function:"):], "new_code": new_code}
        )
    if region.startswith("kernel:"):
        return _op_replace_function(
            text, {"function_name": region[len("kernel:"):], "new_code": new_code}
        )
    raise PatchError(f"unsupported region: {region!r}; supported: function:NAME, kernel:NAME")


def _op_insert_after_region(text: str, op: dict[str, Any]) -> tuple[str, str]:
    region = op["region"]
    new_code = op["new_code"].rstrip("\n") + "\n"
    if region.startswith(("function:", "kernel:")):
        name = region.split(":", 1)[1]
        tree = ast.parse(text)
        fn = _find_top_level_function(tree, name)
        _, end_line = _function_full_span(fn)
        lines = text.splitlines(keepends=True)
        # Ensure there's a blank line before the insert for readability.
        prefix = "" if lines[end_line - 1 : end_line] and lines[end_line - 1].endswith("\n\n") else "\n"
        new_text = "".join(lines[:end_line]) + prefix + new_code + "".join(lines[end_line:])
        return new_text, f"after line {end_line}"
    raise PatchError(f"unsupported region for insert: {region!r}")


def _op_regex_replace(text: str, op: dict[str, Any]) -> tuple[str, str]:
    pattern = op["pattern"]
    replacement = op["replacement"]
    count = op.get("count", 0)
    new_text, n = re.subn(pattern, replacement, text, count=count, flags=re.MULTILINE | re.DOTALL)
    if n == 0:
        raise PatchError(f"regex matched nothing: {pattern!r}")
    return new_text, f"{n} substitution(s)"


def _op_full_file_replace(text: str, op: dict[str, Any]) -> tuple[str, str]:
    new_code = op["new_code"]
    return new_code, "full file replaced"


_OP_HANDLERS: dict[str, Callable[[str, dict[str, Any]], tuple[str, str]]] = {
    "update_constant": _op_update_constant,
    "update_kernel_meta": _op_update_kernel_meta,
    "replace_function": _op_replace_function,
    "replace_kernel_body": _op_replace_kernel_body,
    "replace_region": _op_replace_region,
    "insert_after_region": _op_insert_after_region,
    "regex_replace": _op_regex_replace,
    "full_file_replace": _op_full_file_replace,
}
