from __future__ import annotations

from typing import Any, TypedDict


class TritonDeltaState(TypedDict, total=False):
    base_onnx: str
    target_onnx: str
    triton_model_dir: str
    out_dir: str
    model_name: str
    log_dir: str
    run_id: str
    delta_report: dict[str, Any]
    compact_delta: dict[str, Any]
    triton_context: dict[str, Any]
    patch_plan: dict[str, Any]
    patch_ops: list[dict[str, Any]]
    patch_results: list[dict[str, Any]]
    token_usage: list[dict[str, Any]]
    llm_review: str
    result: dict[str, Any]
    errors: list[str]
