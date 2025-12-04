from langgraph.graph import StateGraph, END

from .nodes import (
    RAGState,
    classify_query,
    retrieve_docs,
    inspect_retrieval,
    answer_with_docs,
    chitchat_answer,
    fallback_answer,
    finalize_answer,
)


def route_after_classify(state: RAGState) -> str:
    """
    Decide which node to go to after classify_query.
    """
    intent = state.get("intent", "rag")
    if intent == "chitchat":
        return "chitchat_answer"
    if intent == "off_topic":
        return "fallback_answer"
    # default: rag
    return "retrieve_docs"


def route_after_inspect(state: RAGState) -> str:
    """
    Decide which node to go to after inspect_retrieval.
    """
    status = state.get("retrieval_status", "weak")
    if status == "ok":
        return "answer_with_docs"
    return "fallback_answer"


def build_graph():
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("inspect_retrieval", inspect_retrieval)
    graph.add_node("answer_with_docs", answer_with_docs)
    graph.add_node("chitchat_answer", chitchat_answer)
    graph.add_node("fallback_answer", fallback_answer)
    graph.add_node("finalize_answer", finalize_answer)

    # Entry point
    graph.set_entry_point("classify_query")

    # Conditional edge after classify_query
    graph.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "retrieve_docs": "retrieve_docs",
            "chitchat_answer": "chitchat_answer",
            "fallback_answer": "fallback_answer",
        },
    )

    # Retrieval path
    graph.add_edge("retrieve_docs", "inspect_retrieval")

    graph.add_conditional_edges(
        "inspect_retrieval",
        route_after_inspect,
        {
            "answer_with_docs": "answer_with_docs",
            "fallback_answer": "fallback_answer",
        },
    )

    # All answer nodes go to finalizer
    graph.add_edge("answer_with_docs", "finalize_answer")
    graph.add_edge("chitchat_answer", "finalize_answer")
    graph.add_edge("fallback_answer", "finalize_answer")

    # End
    graph.add_edge("finalize_answer", END)

    return graph.compile()


graph = build_graph()
