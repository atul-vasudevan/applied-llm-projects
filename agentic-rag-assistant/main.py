from dotenv import load_dotenv
load_dotenv()

from src.graph import graph
from src.nodes import RAGState


def run_cli():
    print("== Agentic RAG Notes Assistant ==")
    question = input("Ask a question (or 'q' to quit):\n> ").strip()
    if not question or question.lower() == "q":
        return

    initial_state: RAGState = {
        "question": question,
        "intent": "",
        "retrieved_docs": [],
        "retrieval_status": "",
        "answer": "",
        "debug": {},
    }

    final_state = graph.invoke(initial_state)

    print("\n=== ANSWER ===")
    print(final_state.get("answer"))


if __name__ == "__main__":
    while True:
        run_cli()
        print()
