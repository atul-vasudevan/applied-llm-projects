from langgraph.graph import StateGraph, END
from .nodes import (
    State,
    classify_task,
    summarize_text,
    bullet_points,
    analyze_sentiment,
    finalize_answer,
)


def _route_after_classification(state: State) -> str:
    """
    Decide which node to go to after `classify_task`.
    """
    task = state.get("task", "summarize")
    if task == "bullet_points":
        return "bullet_points"
    if task == "sentiment":
        return "analyze_sentiment"
    return "summarize_text"


def build_graph():
    # Create a graph with our State type.
    graph = StateGraph(State)

    # Register nodes by name -> function
    graph.add_node("classify_task", classify_task)
    graph.add_node("summarize_text", summarize_text)
    graph.add_node("bullet_points", bullet_points)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("finalize_answer", finalize_answer)

    # srarting node
    graph.set_entry_point("classify_task")

    # Conditional routing from classifier node
    graph.add_conditional_edges(
        "classify_task",
        _route_after_classification,
        {
            "summarize_text": "summarize_text",
            "bullet_points": "bullet_points",
            "analyze_sentiment": "analyze_sentiment",
        },
    )

    # After any processing node, always go to finalizer
    graph.add_edge("summarize_text", "finalize_answer")
    graph.add_edge("bullet_points", "finalize_answer")
    graph.add_edge("analyze_sentiment", "finalize_answer")

    # End after finalizer
    graph.add_edge("finalize_answer", END)

    # Compile into a runnable object
    return graph.compile()


graph = build_graph()
