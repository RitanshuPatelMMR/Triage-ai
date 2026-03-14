from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    parse_and_clean,
    extract_entities,
    check_drug_interactions,
    rag_enrich,
    generate_summary
)


def build_graph():
    """Build and compile the 5-node LangGraph agent"""

    graph = StateGraph(AgentState)

    # ── Add all 5 nodes ───────────────────────────────
    graph.add_node("parse_and_clean", parse_and_clean)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("check_drug_interactions", check_drug_interactions)
    graph.add_node("rag_enrich", rag_enrich)
    graph.add_node("generate_summary", generate_summary)

    # ── Wire nodes in sequence ────────────────────────
    graph.set_entry_point("parse_and_clean")
    graph.add_edge("parse_and_clean", "extract_entities")
    graph.add_edge("extract_entities", "check_drug_interactions")
    graph.add_edge("check_drug_interactions", "rag_enrich")
    graph.add_edge("rag_enrich", "generate_summary")
    graph.add_edge("generate_summary", END)

    return graph.compile()


# Build once at import time — reused on every request
agent = build_graph()