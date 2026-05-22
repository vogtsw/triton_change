from __future__ import annotations

from typing import Any, TypedDict


class TritonDeltaState(TypedDict, total=False):
    base_onnx: str
    target_onnx: str
    triton_model_dir: str
    out_dir: str
    model_name: str
    delta_report: dict[str, Any]
    patch_plan: dict[str, Any]
    llm_review: str
    result: dict[str, Any]
    errors: list[str]

