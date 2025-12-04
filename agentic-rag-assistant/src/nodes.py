from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .loader import build_vectorstore


llm = ChatOllama(model="llama3.1", temperature=0)
vectorstore = build_vectorstore(limit=500)


class RAGState(TypedDict, total=False):
    question: str
    intent: str  # "chitchat" | "rag" | "off_topic"
    retrieved_docs: List[Document]
    retrieval_status: str  # "ok" | "weak" (decided by inspect_retrieval)
    answer: str
    debug: Dict[str, Any]


def classify_query(state: RAGState) -> RAGState:
    """
    Decide whether this is:
    - chitchat      -> answer without retrieval
    - rag           -> answer using the corpus
    - off_topic     -> politely decline
    """
    question = state["question"]
    system = SystemMessage(
        content="""You are a router for an assistant that can answer questions using "
            "a small news/article corpus.\n\n"
            "Possible intents:\n"
            "- chitchat: casual conversation, opinions, generic AI chit-chat.\n"
            "- rag: factual questions that could be grounded in articles.\n"
            "- off_topic: anything clearly unrelated or unsafe.\n\n"
            "Return ONLY one word: chitchat, rag, or off_topic."""
    )
    human = HumanMessage(content=f"User question: {question}")
    resp = llm.invoke([system, human])
    intent_raw = resp.content.strip().lower()

    if "chitchat" in intent_raw:
        intent = "chitchat"
    elif "rag" in intent_raw:
        intent = "rag"
    else:
        intent = "off_topic"

    debug = dict(state.get("debug", {}))
    debug["router_raw"] = intent_raw
    return {
        "intent": intent,
        "debug": debug,
    }


def retrieve_docs(state: RAGState) -> RAGState:
    """
    Retrieve top-k similar docs from the vector store.
    We’ll also stash scores in debug for 'inspect_retrieval' to look at.
    """
    question = state["question"]
    results = vectorstore.similarity_search_with_score(question, k=4)

    docs = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]

    debug = dict(state.get("debug", {}))
    debug["retrieval_scores"] = scores

    return {
        "retrieved_docs": docs,
        "debug": debug,
    }


def inspect_retrieval(state: RAGState) -> RAGState:
    """
    Look at retrieved docs and decide:
    - retrieval_status = "ok"   (good enough context)
    - retrieval_status = "weak" (nothing or poor matches)
    """
    docs = state.get("retrieved_docs") or []
    scores = state.get("debug", {}).get("retrieval_scores", [])

    if not docs:
        status = "weak"
    else:
        status = "ok" if len(docs) >= 1 else "weak"

    debug = dict(state.get("debug", {}))
    debug["retrieval_status"] = status

    return {
        "retrieval_status": status,
        "debug": debug,
    }


def answer_with_docs(state: RAGState) -> RAGState:
    """
    Use retrieved docs as context and answer with citations.
    """
    question = state["question"]
    docs = state.get("retrieved_docs") or []

    context_blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"doc_{idx}")
        context_blocks.append(f"[{idx}] (source={source})\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)

    system = SystemMessage(
        content=(
            "You are an assistant that answers questions using the provided context. "
            "If the context is relevant, answer concisely and refer to sources using "
            "inline citations like [1], [2]. If the context does not contain enough "
            "information, say you don't know rather than inventing details."
        )
    )
    human = HumanMessage(
        content=(
            f"Question:\n{question}\n\n"
            f"Context (numbered sources):\n{context_text}"
        )
    )
    resp = llm.invoke([system, human])

    return {
        "answer": resp.content,
    }


def chitchat_answer(state: RAGState) -> RAGState:
    """
    Simple direct answer without retrieval.
    """
    question = state["question"]

    system = SystemMessage(
        content="You are a friendly general-purpose assistant for casual chat."
    )
    human = HumanMessage(content=question)
    resp = llm.invoke([system, human])

    return {
        "answer": resp.content,
    }


def fallback_answer(state: RAGState) -> RAGState:
    """
    Fallback when query is off-topic or retrieval is weak.
    """
    question = state["question"]
    msg = (
        "I'm not able to answer this question with the knowledge base I have. "
        "It might be outside the scope of the dataset I'm using."
    )

    return {
        "answer": msg,
    }


def finalize_answer(state: RAGState) -> RAGState:
    """
    Attach a small header indicating mode + retrieval status.
    """
    intent = state.get("intent", "unknown")
    retrieval_status = state.get("retrieval_status", "n/a")
    header = f"[mode: {intent}, retrieval: {retrieval_status}]\n"
    body = state.get("answer", "")

    return {
        "answer": header + body
    }