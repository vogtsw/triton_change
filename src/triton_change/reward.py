"""Reward calculator per spec section 6.

Pure function over tool results — no I/O, no LLM. Each component is independently
visible in `RewardBreakdown.components` so that anti-hacking analysis can track
per-component drift over time.

Reward composition (matches spec):

  Main (dense):
    syntax_pass                 +0.10
    import_pass                 +0.10
    model_forward_callable      +0.20
    output_shape_match          +0.30
    output_dtype_match          +0.20
    numerical_correctness_pass  +1.00
    benchmark_no_regression     +0.20  (only if correctness passed)

  Conditional (gated):
    small_localized_patch       +0.10  (gate: shape_match)
    semantic_label_addressed    +0.20  (gate: at least one matching op target)

  Penalties:
    syntax_error                -0.40 (terminal: short-circuits)
    import_error                -0.30
    runtime_error               -0.30
    shape_mismatch              -0.50
    numerical_mismatch          -0.70
    unsafe_code                 -1.00 (terminal)
    oversized_patch             -0.30
    timeout                     -1.00 (terminal)
    patch_apply_error           -0.30 (terminal)
    repeated_same_error         -0.20

  Final all-or-nothing bonus:
    +0.50 if (correctness pass) AND (patch_count <= 5) AND (no unsafe code)

Oracle task_000001 (correctness pass, 2 surgical update_constant ops, 1
matching semantic label) is expected to land at:

    syntax(0.10) + import(0.10) + callable(0.20) + shape(0.30) + dtype(0.20)
  + numerical(1.00) + small(0.10) + semantic(0.20) + bonus(0.50) = 2.70
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


__all__ = ["RewardBreakdown", "compute_reward"]


@dataclass
class RewardBreakdown:
    components: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    failure_class: str | None = None
    success: bool = False  # = numerical correctness pass

    def add(self, name: str, value: float) -> None:
        self.components[name] = self.components.get(name, 0.0) + value
        self.total = round(sum(self.components.values()), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "total": self.total,
            "failure_class": self.failure_class,
            "success": self.success,
        }


def compute_reward(
    static_result: dict | None,
    correctness_result: dict | None,
    patch_ops: list[dict],
    semantic_labels: list[dict],
    *,
    benchmark_result: dict | None = None,
    timed_out: bool = False,
    repeated_same_error: bool = False,
    patch_apply_error: bool = False,
) -> RewardBreakdown:
    rb = RewardBreakdown()

    # Terminal: patch couldn't even be applied
    if patch_apply_error:
        rb.add("patch_apply_error", -0.30)
        rb.failure_class = "patch_apply_error"
        return rb

    # Terminal: timeout
    if timed_out:
        rb.add("timeout", -1.00)
        rb.failure_class = "timeout"
        return rb

    if static_result is None:
        return rb

    # ---- Static check rewards / penalties ----
    if static_result.get("syntax_ok"):
        rb.add("syntax_pass", 0.10)
    else:
        rb.add("syntax_error", -0.40)
        rb.failure_class = "syntax"
        return rb  # short-circuit: nothing else useful to evaluate

    if static_result.get("imports_ok"):
        rb.add("import_pass", 0.10)

    if static_result.get("danger_findings"):
        rb.add("unsafe_code", -1.00)
        rb.failure_class = "unsafe_code"
        return rb

    if correctness_result is None:
        return rb

    cr = correctness_result
    if not cr.get("executed"):
        return rb

    fc = cr.get("failure_class")

    if fc == "import":
        rb.add("import_error", -0.30)
        rb.failure_class = "import"
        return rb

    if fc == "runtime":
        rb.add("runtime_error", -0.30)
        rb.failure_class = "runtime"
        # we still let later checks run because runtime can mean shape/dtype
        # discoverable from the trace; but for clean MVP we short-circuit:
        return rb

    if fc == "timeout":
        rb.add("timeout", -1.00)
        rb.failure_class = "timeout"
        return rb

    # If it executed at all, the model_forward is callable.
    rb.add("model_forward_callable", 0.20)

    if cr.get("shape_match"):
        rb.add("output_shape_match", 0.30)
    else:
        rb.add("shape_mismatch", -0.50)
        rb.failure_class = rb.failure_class or "shape_mismatch"

    if cr.get("dtype_match"):
        rb.add("output_dtype_match", 0.20)

    if cr.get("passed"):
        rb.add("numerical_correctness_pass", 1.00)
        rb.success = True
    elif fc == "numerical_diverge":
        rb.add("numerical_mismatch", -0.70)
        rb.failure_class = rb.failure_class or "numerical_diverge"

    # ---- Conditional bonuses ----
    if cr.get("shape_match") and _is_small_patch(patch_ops):
        rb.add("small_localized_patch", 0.10)

    if _addresses_semantic_labels(patch_ops, semantic_labels):
        rb.add("semantic_label_addressed", 0.20)

    # ---- Patch-quality penalties ----
    if _is_oversized_patch(patch_ops):
        rb.add("oversized_patch", -0.30)

    if repeated_same_error:
        rb.add("repeated_same_error", -0.20)

    # ---- Optional benchmark (gated on correctness) ----
    if cr.get("passed") and benchmark_result and benchmark_result.get("no_regression"):
        rb.add("benchmark_no_regression", 0.20)

    # ---- Final all-or-nothing bonus ----
    if (
        cr.get("passed")
        and len(patch_ops) <= 5
        and not static_result.get("danger_findings")
    ):
        rb.add("all_or_nothing_bonus", 0.50)

    return rb


# ---------- Heuristics ----------


_SURGICAL_OPS = {"update_constant", "update_kernel_meta"}


def _is_small_patch(ops: list[dict]) -> bool:
    """Small = <= 4 ops, all surgical (constant or kernel meta)."""
    if not ops:
        # zero patches is fine when truly nothing needed; still small.
        return True
    return len(ops) <= 4 and all(op.get("operation") in _SURGICAL_OPS for op in ops)


def _is_oversized_patch(ops: list[dict]) -> bool:
    if any(op.get("operation") == "full_file_replace" for op in ops):
        return True
    return len(ops) > 8


def _addresses_semantic_labels(ops: list[dict], labels: list[dict]) -> bool:
    if not labels:
        return False
    op_targets: set[str] = set()
    for op in ops:
        kind = op.get("operation")
        if kind == "update_constant":
            op_targets.add(f"constant:{op.get('constant_name')}")
        elif kind == "update_kernel_meta":
            op_targets.add(f"kernel_meta:{op.get('kernel_name')}.{op.get('meta_name')}")
        elif kind == "replace_kernel_body":
            op_targets.add(f"kernel:{op.get('kernel_name')}")
        elif kind == "replace_function":
            op_targets.add(f"function:{op.get('function_name')}")
        elif kind == "replace_region":
            op_targets.add(op.get("region", ""))
    for label in labels:
        hint = label.get("affected_region_hint", "")
        for piece in hint.split(","):
            piece = piece.strip()
            if piece and piece in op_targets:
                return True
    return False
