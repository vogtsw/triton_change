from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from triton_change.onnx_delta.analyzer import analyze_onnx
from triton_change.onnx_delta.schema import DeltaReport, GraphChange, GraphSummary, NodeInfo, TensorInfo


def diff_onnx(base_path: str | Path, target_path: str | Path) -> DeltaReport:
    base = analyze_onnx(base_path)
    target = analyze_onnx(target_path)
    return diff_summaries(base, target)


def diff_summaries(base: GraphSummary, target: GraphSummary) -> DeltaReport:
    changes: list[GraphChange] = []
    changes.extend(_diff_tensor_maps("input", base.inputs, target.inputs))
    changes.extend(_diff_tensor_maps("output", base.outputs, target.outputs))
    changes.extend(_diff_tensor_maps("initializer", base.initializers, target.initializers, initializer=True))
    changes.extend(_diff_op_counts(base.op_counts, target.op_counts))
    changes.extend(_diff_node_sequence(base.nodes, target.nodes))
    if base.architecture_tags != target.architecture_tags:
        changes.append(
            GraphChange(
                category="architecture_tags",
                key="tags",
                before=base.architecture_tags,
                after=target.architecture_tags,
                impact="Model architecture pattern changed.",
                recommendation="Review Triton preprocessing/postprocessing and backend assumptions for the added or removed architecture tags.",
            )
        )
    return DeltaReport(base=base, target=target, changes=changes)


def _diff_tensor_maps(
    category: str,
    base: dict[str, TensorInfo],
    target: dict[str, TensorInfo],
    initializer: bool = False,
) -> list[GraphChange]:
    changes: list[GraphChange] = []
    for name in sorted(base.keys() | target.keys()):
        before = base.get(name)
        after = target.get(name)
        if before is None or after is None:
            changes.append(
                GraphChange(
                    category=category,
                    key=name,
                    before=asdict(before) if before else None,
                    after=asdict(after) if after else None,
                    impact=f"{category} {'added' if after else 'removed'}.",
                    recommendation="Update Triton config and model metadata if this tensor is part of the serving contract.",
                )
            )
            continue
        if before.dtype != after.dtype or before.shape != after.shape:
            impact = "Serving contract changed." if category in {"input", "output"} else "Parameter layout or precision changed."
            recommendation = (
                "Patch config.pbtxt dims/data_type and validate client payload shape."
                if category in {"input", "output"}
                else "Replace model.onnx and check backend code for hardcoded tensor names, shapes, or dtype casts."
            )
            changes.append(
                GraphChange(
                    category=category,
                    key=name,
                    before=asdict(before),
                    after=asdict(after),
                    impact=impact,
                    recommendation=recommendation,
                )
            )
    return changes


def _diff_op_counts(base: dict[str, int], target: dict[str, int]) -> list[GraphChange]:
    changes: list[GraphChange] = []
    before_counts = Counter(base)
    after_counts = Counter(target)
    for op_type in sorted(before_counts.keys() | after_counts.keys()):
        if before_counts[op_type] != after_counts[op_type]:
            changes.append(
                GraphChange(
                    category="op_count",
                    key=op_type,
                    before=before_counts[op_type],
                    after=after_counts[op_type],
                    impact=f"Operator count for {op_type} changed.",
                    recommendation=_op_recommendation(op_type),
                )
            )
    return changes


def _diff_node_sequence(base_nodes: list[NodeInfo], target_nodes: list[NodeInfo]) -> list[GraphChange]:
    changes: list[GraphChange] = []
    base_ops = [node.op_type for node in base_nodes]
    target_ops = [node.op_type for node in target_nodes]
    if base_ops != target_ops:
        changes.append(
            GraphChange(
                category="node_sequence",
                key="op_type_sequence",
                before=base_ops[:80],
                after=target_ops[:80],
                impact="Execution graph topology changed.",
                recommendation="Prefer replacing only model.onnx when Triton backend is generic; patch Python preprocessing if it assumes fixed operator order.",
            )
        )
    return changes


def _op_recommendation(op_type: str) -> str:
    if op_type == "Cast":
        return "Check precision-sensitive paths and make sure backend/client dtype conversion is intentional."
    if op_type == "Conv":
        return "Check Conv channel dimensions; backend code may need updated output metadata if exposed."
    if op_type in {"MatMul", "Gemm"}:
        return "Check linear projection dimensions and output tensor shapes."
    if op_type == "Reshape":
        return "Review shape constants; Triton config dims may need updating."
    return "Replace model.onnx and run a CPU inference smoke test."


def report_to_dict(report: DeltaReport) -> dict[str, Any]:
    return asdict(report)


def compact_report(report: DeltaReport) -> dict[str, Any]:
    return {
        "base": report.base.path,
        "target": report.target.path,
        "base_tags": report.base.architecture_tags,
        "target_tags": report.target.architecture_tags,
        "base_op_counts": report.base.op_counts,
        "target_op_counts": report.target.op_counts,
        "changes": [_compact_change(change) for change in report.changes],
    }


def _compact_change(change: GraphChange) -> dict[str, Any]:
    return compact_change_dict(asdict(change))


def compact_change_dict(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    if item.get("category") == "node_sequence":
        before = item.get("before", [])
        after = item.get("after", [])
        item["before_len"] = len(before)
        item["after_len"] = len(after)
        item["before"] = before[:12]
        item["after"] = after[:12]
    return item
