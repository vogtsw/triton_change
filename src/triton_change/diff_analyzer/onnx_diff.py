"""ONNX diff analyzer — raw structural diff + semantic change labels.

Reads ``base.onnx`` and ``target.onnx`` and produces the two-layer diff
described in spec section 5.1. No protobuf bytes in output; JSON only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


__all__ = ["analyze_onnx_pair", "infer_semantic_labels", "load_diff_from_task"]


def _elem_type_name(elem_type: int) -> str:
    mapping = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.BFLOAT16: "bfloat16",
        TensorProto.DOUBLE: "float64",
        TensorProto.INT64: "int64",
        TensorProto.INT32: "int32",
    }
    return mapping.get(elem_type, f"tensor_type_{elem_type}")


def _shape_of_tensor_type(tt) -> list[int | str]:
    if tt is None or not tt.HasField("shape"):
        return []
    out: list[int | str] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            out.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            out.append(d.dim_param)
        else:
            out.append("?")
    return out


def _io_specs(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    g = model.graph
    specs: dict[str, dict[str, Any]] = {}
    for vi in list(g.input) + list(g.output):
        tt = vi.type.tensor_type
        specs[vi.name] = {
            "name": vi.name,
            "shape": _shape_of_tensor_type(tt),
            "dtype": _elem_type_name(tt.elem_type),
        }
    return specs


def _op_histogram(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _op_count_delta(base: dict[str, int], target: dict[str, int]) -> dict[str, int]:
    keys = set(base) | set(target)
    return {k: target.get(k, 0) - base.get(k, 0) for k in keys if target.get(k, 0) != base.get(k, 0)}


def _attribute_changes(base: onnx.ModelProto, target: onnx.ModelProto) -> list[dict[str, Any]]:
    base_ln = [n for n in base.graph.node if n.op_type == "LayerNormalization"]
    tgt_ln = [n for n in target.graph.node if n.op_type == "LayerNormalization"]
    changes: list[dict[str, Any]] = []
    if base_ln and tgt_ln:
        def _eps(node):
            for a in node.attribute:
                if a.name == "epsilon":
                    return float(a.f)
            return None
        e0, e1 = _eps(base_ln[0]), _eps(tgt_ln[0])
        if e0 is not None and e1 is not None and e0 != e1:
            changes.append({"op_type": "LayerNormalization", "field": "epsilon", "old": e0, "new": e1})
    return changes


def _shape_changes(base_io: dict, target_io: dict, *, kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, b in base_io.items():
        if name not in target_io:
            continue
        t = target_io[name]
        if b["shape"] != t["shape"]:
            out.append({"name": name, "old_shape": b["shape"], "new_shape": t["shape"], "kind": kind})
    return out


def _dtype_changes(base_io: dict, target_io: dict, *, kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, b in base_io.items():
        if name not in target_io:
            continue
        t = target_io[name]
        if b["dtype"] != t["dtype"]:
            out.append({"name": name, "old_dtype": b["dtype"], "new_dtype": t["dtype"], "kind": kind})
    return out


def _activation_changes(base_ops: dict[str, int], target_ops: dict[str, int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    gelu_b, gelu_t = base_ops.get("Gelu", 0), target_ops.get("Gelu", 0)
    relu_b, relu_t = base_ops.get("Relu", 0), target_ops.get("Relu", 0)
    if gelu_b > gelu_t and relu_t > relu_b:
        out.append({"from": "GELU", "to": "ReLU"})
    # BiasGelu often shows as Gelu + Add pattern; heuristic only
    add_delta = target_ops.get("Add", 0) - base_ops.get("Add", 0)
    if gelu_b == gelu_t and add_delta > 0:
        out.append({"from": "GELU", "to": "BiasGELU"})
    return out


def _batch_changes(base_io: dict, target_io: dict) -> list[dict[str, Any]]:
    for name in base_io:
        if name not in target_io:
            continue
        bs, ts = base_io[name]["shape"], target_io[name]["shape"]
        if bs and ts and isinstance(bs[0], int) and ts[0] in ("batch", "B", "N"):
            return [{"from": f"fixed_{bs[0]}", "to": "dynamic"}]
    return []


def infer_semantic_labels(raw_diff: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if raw_diff.get("input_shape_changes") or raw_diff.get("output_shape_changes"):
        labels.append({
            "label": "shape_param_change",
            "from": "onnx_io_shapes",
            "to": "onnx_io_shapes",
            "affected_region_hint": "constants / model_forward",
            "confidence": 0.9,
        })
    if raw_diff.get("input_dtype_changes") or raw_diff.get("output_dtype_changes"):
        labels.append({
            "label": "dtype_change",
            "from": "onnx_dtype",
            "to": "onnx_dtype",
            "affected_region_hint": "constant:DTYPE_NAME",
            "confidence": 0.9,
        })
    for ch in raw_diff.get("attribute_changes", []):
        if ch.get("field") == "epsilon":
            labels.append({
                "label": "norm_change",
                "from": {"ln_eps": ch.get("old")},
                "to": {"ln_eps": ch.get("new")},
                "affected_region_hint": "constant:LN_EPS",
                "confidence": 0.95,
            })
    for ch in raw_diff.get("activation_changes", []):
        labels.append({
            "label": "activation_change",
            "from": {"activation": ch.get("from")},
            "to": {"activation": ch.get("to")},
            "affected_region_hint": "kernel:gelu_act_kernel",
            "confidence": 0.8,
        })
    for ch in raw_diff.get("batch_changes", []):
        labels.append({
            "label": "shape_param_change",
            "from": {"batch": ch.get("from")},
            "to": {"batch": ch.get("to")},
            "affected_region_hint": "function:model_forward",
            "confidence": 0.85,
        })
    if not labels:
        labels.append({
            "label": "unknown",
            "from": None,
            "to": None,
            "affected_region_hint": "",
            "confidence": 0.0,
        })
    return labels


def analyze_onnx_pair(base_path: Path | str, target_path: Path | str) -> dict[str, Any]:
    base_path, target_path = Path(base_path), Path(target_path)
    base = onnx.load(str(base_path))
    target = onnx.load(str(target_path))

    base_io = _io_specs(base)
    tgt_io = _io_specs(target)
    base_ops = _op_histogram(base)
    tgt_ops = _op_histogram(target)

    inputs = {k: v for k, v in base_io.items() if k in {i.name for i in base.graph.input}}
    tgt_inputs = {k: v for k, v in tgt_io.items() if k in {i.name for i in target.graph.input}}
    outputs = {k: v for k, v in base_io.items() if k in {o.name for o in base.graph.output}}
    tgt_outputs = {k: v for k, v in tgt_io.items() if k in {o.name for o in target.graph.output}}

    raw_diff: dict[str, Any] = {
        "input_shape_changes": _shape_changes(inputs, tgt_inputs, kind="input"),
        "output_shape_changes": _shape_changes(outputs, tgt_outputs, kind="output"),
        "input_dtype_changes": _dtype_changes(inputs, tgt_inputs, kind="input"),
        "output_dtype_changes": _dtype_changes(outputs, tgt_outputs, kind="output"),
        "op_count_delta": _op_count_delta(base_ops, tgt_ops),
        "attribute_changes": _attribute_changes(base, target),
        "activation_changes": _activation_changes(base_ops, tgt_ops),
        "batch_changes": _batch_changes(inputs, tgt_inputs),
    }
    semantic = infer_semantic_labels(raw_diff)
    summary_parts = []
    if raw_diff["input_shape_changes"]:
        s = raw_diff["input_shape_changes"][0]
        summary_parts.append(f"input shape {s['old_shape']} -> {s['new_shape']}")
    if raw_diff["input_dtype_changes"]:
        d = raw_diff["input_dtype_changes"][0]
        summary_parts.append(f"dtype {d['old_dtype']} -> {d['new_dtype']}")
    if raw_diff["attribute_changes"]:
        summary_parts.append("LayerNorm epsilon changed")
    if raw_diff["activation_changes"]:
        a = raw_diff["activation_changes"][0]
        summary_parts.append(f"{a['from']} -> {a['to']}")

    return {
        "raw_diff": raw_diff,
        "semantic_labels": semantic,
        "summary_text": "; ".join(summary_parts) or "ONNX graphs differ",
    }


def load_diff_from_task(task_dir: Path | str) -> dict[str, Any]:
    task_dir = Path(task_dir)
    base = task_dir / "base.onnx"
    target = task_dir / "target.onnx"
    if base.exists() and target.exists():
        return analyze_onnx_pair(base, target)
    return json.loads((task_dir / "oracle" / "diff_summary.json").read_text(encoding="utf-8"))
