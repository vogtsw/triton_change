"""Oracle-based patch evaluation (no LLM / no GPU required)."""
from __future__ import annotations

import ast
import copy
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triton_change.patcher import apply_patch_ops


__all__ = [
    "PatchJudgeResult",
    "normalize_ops",
    "compare_patch_ops",
    "ast_matches_oracle",
    "judge_patch_for_task",
    "judge_patch_ops",
]


@dataclass
class PatchJudgeResult:
    task_id: str
    patch_valid: bool
    ops_match_oracle: bool
    ast_match_oracle: bool
    apply_ok: bool
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.apply_ok and self.ast_match_oracle

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "patch_valid": self.patch_valid,
            "ops_match_oracle": self.ops_match_oracle,
            "ast_match_oracle": self.ast_match_oracle,
            "apply_ok": self.apply_ok,
            "correct": self.correct,
            "error": self.error,
            "details": self.details,
        }


def _strip_docstring(tree: ast.Module) -> ast.Module:
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body = tree.body[1:]
    return tree


def _ast_dump(source: str) -> str:
    tree = _strip_docstring(ast.parse(source))
    return ast.dump(tree, include_attributes=False)


def normalize_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop reason/path noise; sort for order-insensitive comparison."""
    out: list[dict[str, Any]] = []
    for op in ops:
        clean = {k: v for k, v in op.items() if k not in {"reason", "path"}}
        out.append(clean)
    return sorted(out, key=lambda d: json.dumps(d, sort_keys=True, default=str))


def compare_patch_ops(candidate: list[dict], oracle: list[dict]) -> bool:
    return normalize_ops(candidate) == normalize_ops(oracle)


def ast_matches_oracle(candidate_source: str, oracle_source: str) -> bool:
    return _ast_dump(candidate_source) == _ast_dump(oracle_source)


def judge_patch_ops(
    old_source: str,
    candidate_ops: list[dict[str, Any]],
    oracle_ops: list[dict[str, Any]],
    oracle_new_source: str,
    *,
    task_id: str = "unknown",
    workspace: Path | None = None,
) -> PatchJudgeResult:
    """Apply candidate ops and compare against oracle (ops + AST)."""
    details: dict[str, Any] = {}
    if not candidate_ops:
        return PatchJudgeResult(
            task_id=task_id, patch_valid=False, ops_match_oracle=False,
            ast_match_oracle=False, apply_ok=False, error="empty patch_ops",
        )

    ops_match = compare_patch_ops(candidate_ops, oracle_ops)
    details["ops_match_oracle"] = ops_match

    res = apply_patch_ops(old_source, candidate_ops, workspace=workspace)
    if not res.all_succeeded:
        return PatchJudgeResult(
            task_id=task_id,
            patch_valid=True,
            ops_match_oracle=ops_match,
            ast_match_oracle=False,
            apply_ok=False,
            error=res.error or "patch apply failed",
            details=details,
        )

    patched = res.candidate_text or ""
    if res.candidate_path:
        details["patched_path"] = str(res.candidate_path)
    ast_match = ast_matches_oracle(patched, oracle_new_source)
    details["ast_match_oracle"] = ast_match

    return PatchJudgeResult(
        task_id=task_id,
        patch_valid=True,
        ops_match_oracle=ops_match,
        ast_match_oracle=ast_match,
        apply_ok=True,
        details=details,
    )


def judge_patch_for_task(task_dir: Path, candidate_ops: list[dict[str, Any]]) -> PatchJudgeResult:
    task_dir = Path(task_dir)
    task_id = task_dir.name
    old_src = (task_dir / "old_model_triton.py").read_text(encoding="utf-8")
    oracle_doc = json.loads((task_dir / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))
    oracle_ops = oracle_doc["ops"]
    oracle_new = (task_dir / "oracle" / "new_model_triton.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        return judge_patch_ops(
            old_src, candidate_ops, oracle_ops, oracle_new,
            task_id=task_id, workspace=Path(td),
        )
