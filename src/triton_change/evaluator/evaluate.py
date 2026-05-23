"""Unified task evaluator — patch apply, static, correctness, reward.

Spec section 5 + hidden_eval materials. No LLM. CPU-demo path when Triton absent.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triton_change.correctness import correctness_check
from triton_change.evaluator.patch_judge import PatchJudgeResult, judge_patch_for_task
from triton_change.patcher import apply_patch_ops
from triton_change.reward import RewardBreakdown, compute_reward
from triton_change.static_check import static_check


__all__ = ["EvaluatorResult", "evaluate_patch_ops", "evaluate_oracle"]


@dataclass
class EvaluatorResult:
    task_id: str
    patch_apply_ok: bool
    static_passed: bool
    correctness_passed: bool
    oracle_ast_match: bool
    reward: RewardBreakdown
    judge: PatchJudgeResult | None = None
    static_result: dict[str, Any] | None = None
    correctness_result: dict[str, Any] | None = None
    patch_ops: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.correctness_passed or self.oracle_ast_match

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "patch_apply_ok": self.patch_apply_ok,
            "static_passed": self.static_passed,
            "correctness_passed": self.correctness_passed,
            "oracle_ast_match": self.oracle_ast_match,
            "success": self.success,
            "reward": self.reward.to_dict(),
            "judge": self.judge.to_dict() if self.judge else None,
            "static_result": self.static_result,
            "correctness_result": self.correctness_result,
            "patch_ops_count": len(self.patch_ops),
        }


def _semantic_labels(task_dir: Path) -> list[dict[str, Any]]:
    p = task_dir / "hidden_eval" / "semantic_change_labels.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc if isinstance(doc, list) else doc.get("labels", [])


def evaluate_patch_ops(
    task_dir: Path | str,
    patch_ops: list[dict[str, Any]],
    *,
    device: str = "cpu",
    timeout: float = 60.0,
    use_cpu_demo: bool = False,
    compare_oracle_ast: bool = True,
) -> EvaluatorResult:
    task_dir = Path(task_dir)
    task_id = task_dir.name
    labels = _semantic_labels(task_dir)
    old_src = (task_dir / "old_model_triton.py").read_text(encoding="utf-8")

    judge = judge_patch_for_task(task_dir, patch_ops) if compare_oracle_ast else None

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        apply_res = apply_patch_ops(old_src, patch_ops, workspace=ws)
        if not apply_res.all_succeeded:
            rb = compute_reward(None, None, patch_ops, labels, patch_apply_error=True)
            return EvaluatorResult(
                task_id=task_id,
                patch_apply_ok=False,
                static_passed=False,
                correctness_passed=False,
                oracle_ast_match=bool(judge and judge.correct),
                reward=rb,
                judge=judge,
                patch_ops=patch_ops,
            )
        cand_text = apply_res.candidate_text
        sandbox = task_dir / "sandbox"
        sandbox.mkdir(exist_ok=True)
        cand = sandbox / "candidate_model_triton.py"
        cand.write_text(cand_text, encoding="utf-8")

    static = static_check(cand)
    static_d = static.to_dict()

    corr_d = None
    passed = False
    if static.passed:
        if use_cpu_demo and judge and judge.correct:
            cpu_patch = task_dir / "oracle" / "cpu_demo_patch_ops.json"
            demo_ops = json.loads(cpu_patch.read_text(encoding="utf-8"))["ops"]
            with tempfile.TemporaryDirectory() as td2:
                demo_res = apply_patch_ops(old_src, demo_ops, workspace=Path(td2))
                eval_cand = demo_res.candidate_path or Path(td2) / "candidate_model_triton.py"
                cr = correctness_check(eval_cand, task_dir, timeout=timeout, device=device)
                corr_d = cr.to_dict()
                passed = cr.passed
        else:
            eval_cand = cand
            cr = correctness_check(eval_cand, task_dir, timeout=timeout, device=device)
            corr_d = cr.to_dict()
            passed = cr.passed

    if judge and judge.correct and use_cpu_demo:
        passed = True
        if corr_d is None:
            corr_d = {"passed": True, "note": "oracle_ast_match"}

    rb = compute_reward(static_d, corr_d, patch_ops, labels)
    if passed and not rb.success:
        rb.success = True

    return EvaluatorResult(
        task_id=task_id,
        patch_apply_ok=True,
        static_passed=static.passed,
        correctness_passed=passed,
        oracle_ast_match=bool(judge and judge.correct),
        reward=rb,
        judge=judge,
        static_result=static_d,
        correctness_result=corr_d,
        patch_ops=patch_ops,
    )


def evaluate_oracle(task_dir: Path | str, **kw: Any) -> EvaluatorResult:
    task_dir = Path(task_dir)
    ops = json.loads((task_dir / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]
    return evaluate_patch_ops(task_dir, ops, compare_oracle_ast=True, **kw)
