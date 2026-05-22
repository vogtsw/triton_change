from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from triton_change.onnx_delta.diff import diff_onnx, report_to_dict
from triton_change.onnx_delta.schema import DeltaReport, TensorInfo
from triton_change.triton.repository import config_pbtxt


ONNX_TO_TRITON_DTYPE = {
    "FLOAT": "TYPE_FP32",
    "FLOAT16": "TYPE_FP16",
    "DOUBLE": "TYPE_FP64",
    "INT64": "TYPE_INT64",
    "INT32": "TYPE_INT32",
    "INT8": "TYPE_INT8",
    "UINT8": "TYPE_UINT8",
    "BOOL": "TYPE_BOOL",
}


@dataclass
class FilePatch:
    path: str
    action: str
    reason: str


@dataclass
class TritonPatchPlan:
    model_name: str
    files: list[FilePatch]
    warnings: list[str]
    triton_config: str
    delta: dict[str, Any]


def create_patch_plan(model_name: str, report: DeltaReport) -> TritonPatchPlan:
    input_info = _first_tensor(report.target.inputs, preferred="input_ids")
    output_info = _first_tensor(report.target.outputs, preferred="logits")
    input_dims = _triton_dims(input_info)
    output_dims = _triton_dims(output_info)
    config = _config_from_tensors(model_name, input_info, output_info, input_dims, output_dims)
    warnings = _warnings(report)
    files = [
        FilePatch("model.onnx", "replace", "The executable graph changed; swap in the target ONNX graph."),
        FilePatch("config.pbtxt", "patch", "Input/output serving contract is regenerated from target ONNX value info."),
        FilePatch("delta_report.json", "create", "Keep a machine-readable audit trail for the graph change."),
    ]
    if any(change.category == "op_count" and change.key == "Cast" for change in report.changes):
        files.append(FilePatch("precision_notes.md", "create", "Document inserted or removed Cast operators for serving review."))
    return TritonPatchPlan(
        model_name=model_name,
        files=files,
        warnings=warnings,
        triton_config=config,
        delta=report_to_dict(report),
    )


def apply_incremental_patch(
    triton_model_dir: Path,
    base_onnx: Path,
    target_onnx: Path,
    out_dir: Path,
    model_name: str | None = None,
) -> TritonPatchPlan:
    triton_model_dir = Path(triton_model_dir)
    out_dir = Path(out_dir)
    model_name = model_name or _model_name_from_config(triton_model_dir / "config.pbtxt") or triton_model_dir.name
    report = diff_onnx(base_onnx, target_onnx)
    plan = create_patch_plan(model_name, report)

    out_dir.mkdir(parents=True, exist_ok=True)
    patched_model_dir = out_dir / model_name
    shutil.copytree(triton_model_dir, patched_model_dir, dirs_exist_ok=True)
    shutil.copy2(target_onnx, patched_model_dir / "model.onnx")
    (patched_model_dir / "config.pbtxt").write_text(plan.triton_config, encoding="utf-8")
    (patched_model_dir / "delta_report.json").write_text(json.dumps(plan.delta, indent=2, ensure_ascii=False), encoding="utf-8")
    if any(file.path == "precision_notes.md" for file in plan.files):
        (patched_model_dir / "precision_notes.md").write_text(_precision_notes(report), encoding="utf-8")
    (out_dir / "patch_plan.json").write_text(json.dumps(asdict(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def _config_from_tensors(
    model_name: str,
    input_info: TensorInfo,
    output_info: TensorInfo,
    input_dims: list[int],
    output_dims: list[int],
) -> str:
    input_dtype = ONNX_TO_TRITON_DTYPE.get(input_info.dtype, "TYPE_INVALID")
    output_dtype = ONNX_TO_TRITON_DTYPE.get(output_info.dtype, "TYPE_INVALID")
    return f'''name: "{model_name}"
backend: "python"
max_batch_size: 0
input [
  {{
    name: "{input_info.name}"
    data_type: {input_dtype}
    dims: [ {", ".join(str(d) for d in input_dims)} ]
  }}
]
output [
  {{
    name: "{output_info.name}"
    data_type: {output_dtype}
    dims: [ {", ".join(str(d) for d in output_dims)} ]
  }}
]
instance_group [
  {{
    kind: KIND_CPU
  }}
]
'''


def _first_tensor(tensors: dict[str, TensorInfo], preferred: str) -> TensorInfo:
    if preferred in tensors:
        return tensors[preferred]
    if not tensors:
        raise ValueError("ONNX graph has no tensors in the requested map")
    return next(iter(tensors.values()))


def _triton_dims(info: TensorInfo) -> list[int]:
    dims: list[int] = []
    for dim in info.shape:
        if isinstance(dim, int):
            dims.append(dim)
        else:
            dims.append(-1)
    return dims


def _model_name_from_config(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    match = re.search(r'name:\s*"([^"]+)"', config_path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _warnings(report: DeltaReport) -> list[str]:
    warnings: list[str] = []
    for change in report.changes:
        if change.category in {"input", "output"}:
            warnings.append(f"{change.category}:{change.key} changed; clients may need payload updates.")
        if change.category == "op_count" and change.key == "Cast":
            warnings.append("Cast operator count changed; verify precision behavior on the target serving backend.")
        if change.category == "initializer" and "conv" in change.key.lower():
            warnings.append(f"{change.key} changed shape or dtype; Conv branch dimensions changed.")
    return sorted(set(warnings))


def _precision_notes(report: DeltaReport) -> str:
    lines = ["# Precision Notes", ""]
    for change in report.changes:
        if change.category == "op_count" and change.key == "Cast":
            lines.append(f"- Cast count changed from `{change.before}` to `{change.after}`.")
    lines.append("- Confirm that client inputs and Triton output metadata still use the intended dtype.")
    return "\n".join(lines) + "\n"

