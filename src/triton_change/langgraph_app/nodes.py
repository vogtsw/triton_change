from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from triton_change.langgraph_app.llm import complete_patch_review
from triton_change.langgraph_app.state import TritonDeltaState
from triton_change.onnx_delta.diff import compact_change_dict, diff_onnx, report_to_dict
from triton_change.triton.patcher import apply_incremental_patch, create_patch_plan


def analyze_graphs(state: TritonDeltaState) -> TritonDeltaState:
    report = diff_onnx(state["base_onnx"], state["target_onnx"])
    return {"delta_report": report_to_dict(report)}


def plan_patch(state: TritonDeltaState) -> TritonDeltaState:
    report = diff_onnx(state["base_onnx"], state["target_onnx"])
    model_name = state.get("model_name") or Path(state["triton_model_dir"]).name
    plan = create_patch_plan(model_name, report)
    return {"patch_plan": asdict(plan)}


def review_with_llm(state: TritonDeltaState) -> TritonDeltaState:
    review = complete_patch_review(state.get("delta_report", {}), state.get("patch_plan", {}))
    return {"llm_review": review}


def apply_patch_node(state: TritonDeltaState) -> TritonDeltaState:
    plan = apply_incremental_patch(
        triton_model_dir=Path(state["triton_model_dir"]),
        base_onnx=Path(state["base_onnx"]),
        target_onnx=Path(state["target_onnx"]),
        out_dir=Path(state["out_dir"]),
        model_name=state.get("model_name"),
    )
    return {"result": asdict(plan)}


def summarize(state: TritonDeltaState) -> TritonDeltaState:
    result = state.get("result", {})
    delta = result.get("delta", {})
    if isinstance(delta, dict) and "changes" in delta:
        result["delta"] = {
            "change_count": len(delta.get("changes", [])),
            "changes": [compact_change_dict(change) for change in delta.get("changes", [])],
        }
    result["llm_review"] = state.get("llm_review", "")
    return {"result": result}
