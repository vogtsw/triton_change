from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: list[str | int]


@dataclass
class NodeInfo:
    index: int
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSummary:
    path: str
    ir_version: int
    opset_imports: dict[str, int]
    inputs: dict[str, TensorInfo]
    outputs: dict[str, TensorInfo]
    value_info: dict[str, TensorInfo]
    initializers: dict[str, TensorInfo]
    nodes: list[NodeInfo]
    op_counts: dict[str, int]
    architecture_tags: list[str]


@dataclass
class GraphChange:
    category: str
    key: str
    before: Any
    after: Any
    impact: str
    recommendation: str


@dataclass
class DeltaReport:
    base: GraphSummary
    target: GraphSummary
    changes: list[GraphChange]

