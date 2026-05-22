from __future__ import annotations

from pathlib import Path

from triton_change.langgraph_app.llm import make_patch_ops
from triton_change.langgraph_app.logging import logger_from_state
from triton_change.langgraph_app.state import TritonDeltaState
from triton_change.onnx_delta.diff import compact_report, diff_onnx, report_to_dict
from triton_change.triton.patch_ops import apply_patch_ops, inspect_triton_model, write_patch_summary
from triton_change.triton.patcher import ONNX_TO_TRITON_DTYPE


def analyze_graphs(state: TritonDeltaState) -> TritonDeltaState:
    logger = logger_from_state(state)
    logger.node_input("analyze_graphs", dict(state))
    report = diff_onnx(state["base_onnx"], state["target_onnx"])
    output = {"delta_report": report_to_dict(report), "compact_delta": compact_report(report)}
    logger.node_output("analyze_graphs", output)
    return output


def inspect_triton_code(state: TritonDeltaState) -> TritonDeltaState:
    logger = logger_from_state(state)
    logger.node_input("inspect_triton_code", dict(state))
    context = inspect_triton_model(Path(state["triton_model_dir"]))
    output = {"triton_context": context}
    logger.node_output("inspect_triton_code", output)
    return output


def plan_patch_ops(state: TritonDeltaState) -> TritonDeltaState:
    logger = logger_from_state(state)
    logger.node_input("plan_patch_ops", dict(state))
    fallback_ops = _fallback_patch_ops(state)
    llm_result = make_patch_ops(
        compact_delta=state.get("compact_delta", {}),
        triton_context=state.get("triton_context", {}),
        fallback_ops=fallback_ops,
    )
    output = {
        "patch_ops": llm_result["patch_ops"],
        "token_usage": [llm_result["usage"]],
        "llm_review": llm_result.get("raw", ""),
    }
    logger.llm_call(
        "make_patch_ops",
        {
            "input": {
                "compact_delta": state.get("compact_delta", {}),
                "triton_context": state.get("triton_context", {}),
                "fallback_ops": fallback_ops,
            },
            "output": {"patch_ops": output["patch_ops"], "raw": output["llm_review"]},
            "usage": llm_result["usage"],
        },
    )
    logger.node_output("plan_patch_ops", output)
    return output


def apply_patch_node(state: TritonDeltaState) -> TritonDeltaState:
    logger = logger_from_state(state)
    logger.node_input("apply_patch_node", dict(state))
    patched_model_dir, results = apply_patch_ops(
        source_model_dir=Path(state["triton_model_dir"]),
        out_dir=Path(state["out_dir"]),
        model_name=state.get("model_name") or Path(state["triton_model_dir"]).name,
        ops=state.get("patch_ops", []),
        target_onnx=Path(state["target_onnx"]),
        delta_report=state.get("delta_report", {}),
    )
    logger.tool_call(
        "apply_patch_ops",
        {
            "action": "applied_patch_ops",
            "patched_model_dir": str(patched_model_dir),
            "ops": state.get("patch_ops", []),
            "results": results,
        },
    )
    output = {"patch_results": results, "result": {"patched_model_dir": str(patched_model_dir)}}
    logger.node_output("apply_patch_node", output)
    return output


def write_summary(state: TritonDeltaState) -> TritonDeltaState:
    logger = logger_from_state(state)
    logger.node_input("write_summary", dict(state))
    patched_model_dir = Path(state["result"]["patched_model_dir"])
    summary_path = write_patch_summary(
        patched_model_dir=patched_model_dir,
        ops=state.get("patch_ops", []),
        results=state.get("patch_results", []),
        delta_changes=state.get("compact_delta", {}).get("changes", []),
        token_usage=state.get("token_usage", []),
    )
    logger.tool_call(
        "write_patch_summary",
        {
            "action": "wrote_markdown_summary",
            "path": str(summary_path),
            "token_usage": state.get("token_usage", []),
        },
    )
    output = {
        "result": {
            **state.get("result", {}),
            "patch_summary": str(summary_path),
            "log_dir": str(logger.dir),
            "patch_ops_count": len(state.get("patch_ops", [])),
            "token_usage": state.get("token_usage", []),
            "changes": state.get("compact_delta", {}).get("changes", []),
        }
    }
    logger.node_output("write_summary", output)
    return output


def _fallback_patch_ops(state: TritonDeltaState) -> list[dict]:
    target = state.get("delta_report", {}).get("target", {})
    inputs = target.get("inputs", {})
    outputs = target.get("outputs", {})
    ops: list[dict] = [{"operation": "copy_target_onnx", "path": "model.onnx"}]

    for tensor in list(inputs.values()) + list(outputs.values()):
        name = tensor["name"]
        dtype = ONNX_TO_TRITON_DTYPE.get(tensor["dtype"], "TYPE_INVALID")
        dims = _dims_for_config(tensor["shape"])
        ops.append(
            {
                "operation": "regex_replace",
                "path": "config.pbtxt",
                "pattern": f'(name:\\s*"{name}"[\\s\\S]*?data_type:\\s*)TYPE_\\w+',
                "replacement": f"\\1{dtype}",
                "count": 1,
                "reason": f"Update Triton dtype for {name}.",
            }
        )
        ops.append(
            {
                "operation": "regex_replace",
                "path": "config.pbtxt",
                "pattern": f'(name:\\s*"{name}"[\\s\\S]*?dims:\\s*\\[)[^\\]]*(\\])',
                "replacement": f"\\1 {', '.join(str(d) for d in dims)} \\2",
                "count": 1,
                "reason": f"Update Triton dims for {name}.",
            }
        )

    model_py = state.get("triton_context", {}).get("files", {}).get("1/model.py", {}).get("preview", "")
    input_tensors = list(inputs.values())
    output_tensors = list(outputs.values())
    if "EXPECTED_SEQUENCE_LENGTH" in model_py and input_tensors:
        seq_dim = _dims_for_config(input_tensors[0]["shape"])[-1]
        ops.append(
            {
                "operation": "regex_replace",
                "path": "1/model.py",
                "pattern": r"EXPECTED_SEQUENCE_LENGTH\s*=\s*\d+",
                "replacement": f"EXPECTED_SEQUENCE_LENGTH = {seq_dim}",
                "count": 1,
                "reason": "Update hardcoded Triton Python backend sequence length guard.",
            }
        )
    if "EXPECTED_NUM_CLASSES" in model_py and output_tensors:
        class_dim = _dims_for_config(output_tensors[0]["shape"])[-1]
        ops.append(
            {
                "operation": "regex_replace",
                "path": "1/model.py",
                "pattern": r"EXPECTED_NUM_CLASSES\s*=\s*\d+",
                "replacement": f"EXPECTED_NUM_CLASSES = {class_dim}",
                "count": 1,
                "reason": "Update hardcoded Triton Python backend output class guard.",
            }
        )

    ops.append({"operation": "write_delta_report", "path": "delta_report.json"})
    return ops


def _dims_for_config(shape: list) -> list[int]:
    dims: list[int] = []
    for dim in shape:
        dims.append(dim if isinstance(dim, int) else -1)
    return dims
