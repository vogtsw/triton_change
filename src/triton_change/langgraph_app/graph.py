from __future__ import annotations

from langgraph.graph import END, StateGraph

from triton_change.langgraph_app.nodes import analyze_graphs, apply_patch_node, plan_patch, review_with_llm, summarize
from triton_change.langgraph_app.state import TritonDeltaState


def build_graph():
    graph = StateGraph(TritonDeltaState)
    graph.add_node("analyze_graphs", analyze_graphs)
    graph.add_node("plan_patch", plan_patch)
    graph.add_node("review_with_llm", review_with_llm)
    graph.add_node("apply_patch", apply_patch_node)
    graph.add_node("summarize", summarize)
    graph.set_entry_point("analyze_graphs")
    graph.add_edge("analyze_graphs", "plan_patch")
    graph.add_edge("plan_patch", "review_with_llm")
    graph.add_edge("review_with_llm", "apply_patch")
    graph.add_edge("apply_patch", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()

