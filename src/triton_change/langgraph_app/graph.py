from __future__ import annotations

from langgraph.graph import END, StateGraph

from triton_change.langgraph_app.nodes import analyze_graphs, apply_patch_node, inspect_triton_code, plan_patch_ops, write_summary
from triton_change.langgraph_app.state import TritonDeltaState


def build_graph():
    graph = StateGraph(TritonDeltaState)
    graph.add_node("analyze_graphs", analyze_graphs)
    graph.add_node("inspect_triton_code", inspect_triton_code)
    graph.add_node("plan_patch_ops", plan_patch_ops)
    graph.add_node("apply_patch", apply_patch_node)
    graph.add_node("write_summary", write_summary)
    graph.set_entry_point("analyze_graphs")
    graph.add_edge("analyze_graphs", "inspect_triton_code")
    graph.add_edge("inspect_triton_code", "plan_patch_ops")
    graph.add_edge("plan_patch_ops", "apply_patch")
    graph.add_edge("apply_patch", "write_summary")
    graph.add_edge("write_summary", END)
    return graph.compile()
