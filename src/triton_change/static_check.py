"""Static checks on candidate Triton single-file code (no execution).

Per spec section 5.5. Runs purely via `ast` and never imports the candidate.
The result is a structured object that downstream reward computation can
consume directly. `passed` is the gate for running correctness_check.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from pathlib import Path


__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "DANGEROUS_CALL_NAMES",
    "DANGEROUS_ATTR_PREFIXES",
    "MAX_FILE_BYTES",
    "StaticCheckResult",
    "static_check",
]


ALLOWED_IMPORT_ROOTS: set[str] = {
    "torch",
    "triton",
    "numpy",
    "math",
    "typing",
    "__future__",
    "itertools",
    "functools",
    "dataclasses",
    "collections",
}


DANGEROUS_CALL_NAMES: set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
}


DANGEROUS_ATTR_PREFIXES: tuple[str, ...] = (
    "os.system",
    "os.popen",
    "os.spawn",
    "os.exec",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "subprocess.",
    "socket.",
    "urllib.",
    "requests.",
    "httpx.",
    "shutil.rmtree",
)


MAX_FILE_BYTES = 500 * 1024  # 500KB


@dataclass
class StaticCheckResult:
    syntax_ok: bool
    syntax_error: str | None
    imports_ok: bool
    bad_imports: list[str]
    model_forward_present: bool
    triton_kernels_found: list[str]
    danger_findings: list[str]
    file_size_ok: bool
    file_size_bytes: int
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.syntax_ok
            and self.imports_ok
            and self.model_forward_present
            and not self.danger_findings
            and self.file_size_ok
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def static_check(file_path: Path | str) -> StaticCheckResult:
    p = Path(file_path)
    size = p.stat().st_size
    file_size_ok = size <= MAX_FILE_BYTES

    text = p.read_text(encoding="utf-8")

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return StaticCheckResult(
            syntax_ok=False,
            syntax_error=f"line {e.lineno}: {e.msg}",
            imports_ok=False,
            bad_imports=[],
            model_forward_present=False,
            triton_kernels_found=[],
            danger_findings=[],
            file_size_ok=file_size_ok,
            file_size_bytes=size,
            warnings=[],
        )

    bad_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root and root not in ALLOWED_IMPORT_ROOTS:
                bad_imports.append(mod)

    model_forward_present = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "model_forward"
        for n in tree.body
    )

    triton_kernels: list[str] = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            for dec in n.decorator_list:
                deco = _decorator_name(dec)
                if deco in {"triton.jit", "jit"}:
                    triton_kernels.append(n.name)
                    break

    danger_findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in DANGEROUS_CALL_NAMES:
                danger_findings.append(f"call to {name} at line {node.lineno}")
                continue
            for prefix in DANGEROUS_ATTR_PREFIXES:
                if name.startswith(prefix):
                    danger_findings.append(f"call to {name} at line {node.lineno}")
                    break

    warnings: list[str] = []
    if not triton_kernels:
        warnings.append("no @triton.jit kernel found (acceptable for trivial cases)")

    return StaticCheckResult(
        syntax_ok=True,
        syntax_error=None,
        imports_ok=not bad_imports,
        bad_imports=bad_imports,
        model_forward_present=model_forward_present,
        triton_kernels_found=triton_kernels,
        danger_findings=danger_findings,
        file_size_ok=file_size_ok,
        file_size_bytes=size,
        warnings=warnings,
    )


# ---------- AST helpers ----------


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return _attribute_name(dec)
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return ""


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _attribute_name(func)
    return ""


def _attribute_name(node: ast.Attribute) -> str:
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))
