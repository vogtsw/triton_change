from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, shape_inference

from triton_change.onnx_delta.schema import GraphSummary, NodeInfo, TensorInfo


TENSOR_TYPES = {value: name.replace("TensorProto.", "") for name, value in TensorProto.DataType.items()}


def _dtype_name(elem_type: int) -> str:
    return TENSOR_TYPES.get(elem_type, f"UNKNOWN_{elem_type}")


def _dim_to_value(dim: onnx.TensorShapeProto.Dimension) -> str | int:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return str(dim.dim_param)
    return "?"


def _tensor_info(value: onnx.ValueInfoProto) -> TensorInfo:
    tensor_type = value.type.tensor_type
    shape = [_dim_to_value(dim) for dim in tensor_type.shape.dim]
    return TensorInfo(name=value.name, dtype=_dtype_name(tensor_type.elem_type), shape=shape)


def _initializer_info(init: onnx.TensorProto) -> TensorInfo:
    return TensorInfo(name=init.name, dtype=_dtype_name(init.data_type), shape=[int(d) for d in init.dims])


def _attribute_value(attr: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in value]
    return value


def analyze_onnx(path: str | Path) -> GraphSummary:
    path = Path(path)
    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    inputs = {vi.name: _tensor_info(vi) for vi in graph.input if vi.name not in {i.name for i in graph.initializer}}
    outputs = {vi.name: _tensor_info(vi) for vi in graph.output}
    value_info = {vi.name: _tensor_info(vi) for vi in graph.value_info}
    initializers = {init.name: _initializer_info(init) for init in graph.initializer}
    nodes = [
        NodeInfo(
            index=i,
            name=node.name or f"{node.op_type}_{i}",
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attributes={attr.name: _attribute_value(attr) for attr in node.attribute},
        )
        for i, node in enumerate(graph.node)
    ]
    op_counts = dict(sorted(Counter(node.op_type for node in nodes).items()))
    tags = _architecture_tags(nodes, initializers)
    opsets = {op.domain or "ai.onnx": op.version for op in model.opset_import}
    return GraphSummary(
        path=str(path),
        ir_version=model.ir_version,
        opset_imports=opsets,
        inputs=inputs,
        outputs=outputs,
        value_info=value_info,
        initializers=initializers,
        nodes=nodes,
        op_counts=op_counts,
        architecture_tags=tags,
    )


def _architecture_tags(nodes: list[NodeInfo], initializers: dict[str, TensorInfo]) -> list[str]:
    op_types = {node.op_type for node in nodes}
    tags: list[str] = []
    conv_nodes = [node for node in nodes if node.op_type == "Conv"]
    if conv_nodes:
        for node in conv_nodes:
            weight = next((initializers.get(inp) for inp in node.inputs if inp in initializers), None)
            if weight and len(weight.shape) == 3:
                tags.append("conv1d")
                break
        else:
            tags.append("conv")
    if {"MatMul", "Softmax"}.issubset(op_types) and ("LayerNormalization" in op_types or "LayerNorm" in op_types):
        tags.append("transformer_attention")
    if "Gemm" in op_types or "MatMul" in op_types:
        tags.append("linear_projection")
    if "Cast" in op_types:
        tags.append("precision_cast")
    return sorted(set(tags))

