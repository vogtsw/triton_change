"""Observation construction and code-summary extraction for the agent.

Per spec section 5.3, the agent's input each step is:

    onnx_diff           raw + semantic labels (from oracle/diff_summary.json
                        in the current MVP — produced by diff_analyzer in prod)
    code_summary        constants / kernels / top-level functions of the
                        current candidate (or `old_model_triton.py` on step 0)
    last_action         the prior action's tool name + ops summary
    last_error          failure_class + truncated traceback
    hint                template-driven suggestion based on failure_class
    remaining_steps     budget left in this rollout

We deliberately keep this purely structural: nothing here calls a model or
runs code. The hint generator is a deterministic function of the tool result.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


__all__ = [
    "CodeSummary",
    "Observation",
    "ConstantInfo",
    "KernelInfo",
    "FunctionInfo",
    "extract_code_summary",
    "failure_hint",
    "error_signature",
]


# ---------- Code summary ----------


@dataclass
class ConstantInfo:
    name: str
    value: Any
    lineno: int


@dataclass
class KernelInfo:
    name: str
    lineno: int
    decorator: str  # "triton.jit" / "jit" / etc.
    args: list[str]


@dataclass
class FunctionInfo:
    name: str
    lineno: int
    args: list[str]


@dataclass
class CodeSummary:
    constants: list[ConstantInfo] = field(default_factory=list)
    kernels: list[KernelInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    file_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "constants": [asdict(c) for c in self.constants],
            "kernels": [asdict(k) for k in self.kernels],
            "functions": [asdict(f) for f in self.functions],
            "imports": list(self.imports),
            "file_lines": self.file_lines,
        }


def extract_code_summary(file_path: Path | str) -> CodeSummary:
    p = Path(file_path)
    text = p.read_text(encoding="utf-8")
    tree = ast.parse(text)

    cs = CodeSummary(file_lines=len(text.splitlines()))

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            cs.constants.append(ConstantInfo(
                name=node.targets[0].id, value=value, lineno=node.lineno,
            ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorator = ""
            for dec in node.decorator_list:
                decorator = _decorator_name(dec)
                if decorator in {"triton.jit", "jit"}:
                    break
            args = [a.arg for a in node.args.args]
            if decorator in {"triton.jit", "jit"}:
                cs.kernels.append(KernelInfo(
                    name=node.name, lineno=node.lineno,
                    decorator=decorator, args=args,
                ))
            else:
                cs.functions.append(FunctionInfo(
                    name=node.name, lineno=node.lineno, args=args,
                ))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                cs.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            cs.imports.append(mod)

    return cs


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        parts: list[str] = []
        cur: ast.expr = dec
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return ""


# ---------- Observation ----------


@dataclass
class Observation:
    step_idx: int
    task_id: str
    onnx_diff: dict[str, Any]
    code_summary: dict[str, Any]
    last_action: dict[str, Any] | None = None
    last_error: dict[str, Any] | None = None
    hint: str | None = None
    remaining_steps: int = 0
    repeated_same_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "task_id": self.task_id,
            "onnx_diff": self.onnx_diff,
            "code_summary": self.code_summary,
            "last_action": self.last_action,
            "last_error": self.last_error,
            "hint": self.hint,
            "remaining_steps": self.remaining_steps,
            "repeated_same_error": self.repeated_same_error,
        }


# ---------- Hint templates ----------


_HINT_BY_CLASS: dict[str, str] = {
    "syntax": (
        "Candidate has a Python syntax error. Re-read the affected region and ensure your "
        "patch produces valid Python — watch indentation, missing colons, unclosed brackets."
    ),
    "import": (
        "Candidate cannot be imported. Allowed import roots are torch, triton, "
        "triton.language, numpy, math, typing, dataclasses, itertools, functools, collections, "
        "__future__. Make sure all referenced names exist after your patch."
    ),
    "runtime": (
        "Runtime error during model_forward. Check input shapes, kernel call sites, "
        "BLOCK_SIZE constants, and any hardcoded dim asserts. The traceback tail is below."
    ),
    "shape_mismatch": (
        "Output shape is wrong. Likely candidates: HIDDEN_SIZE / INTERMEDIATE_SIZE constants, "
        "incorrect reshape, or kernel grid mis-sized. Compare the diff's input/output shapes "
        "with the constants in code_summary."
    ),
    "dtype_mismatch": (
        "Output dtype differs from target. Inspect any explicit .to(dtype), tl.float32 casts, "
        "and the dtype of weights / accumulators."
    ),
    "numerical_diverge": (
        "Output has correct shape/dtype but values differ beyond tolerance. Check: LayerNorm "
        "epsilon, activation formula (GELU exact vs tanh approximation), normalization (LN vs "
        "RMSNorm), and weight layout."
    ),
    "timeout": (
        "model_forward exceeded the timeout. Possible causes: infinite loop in a kernel, "
        "BLOCK_SIZE too small causing too many grid launches, or accidental Python loop."
    ),
    "patch_apply_error": (
        "The previous patch could not be applied (constant_name not found, kernel not found, "
        "regex no match, etc.). Re-check the names against code_summary before retrying."
    ),
    "unsafe_code": (
        "Static check flagged unsafe code (eval, exec, os.system, subprocess, network call). "
        "Remove these entirely — only the import whitelist is allowed."
    ),
}


def failure_hint(failure_class: str | None, repeated: bool = False) -> str | None:
    if not failure_class:
        return None
    base = _HINT_BY_CLASS.get(failure_class, f"Tool reported failure_class={failure_class!r}.")
    if repeated:
        return (
            base
            + " STRATEGY HINT: this exact failure has occurred 2+ times — "
            "switch to a DIFFERENT op type or address a different region of the code."
        )
    return base


def error_signature(tool_result: dict[str, Any]) -> str:
    """A short, stable hash of (failure_class + first 200 chars of error/error_tail).

    Used by the runner to detect repeated identical errors.
    """
    fc = tool_result.get("failure_class") or ""
    err = (tool_result.get("error") or "") + (tool_result.get("error_tail") or "")
    sig = (fc + "|" + err[:200]).encode("utf-8", errors="replace")
    return hashlib.sha1(sig).hexdigest()[:12]


def load_diff_summary(task_dir: Path) -> dict[str, Any]:
    """Read oracle/diff_summary.json. In the production diff_analyzer this would
    be computed live from base.onnx vs target.onnx; for Phase 2 baseline we
    consume the precomputed summary that ships with each task."""
    p = task_dir / "oracle" / "diff_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
