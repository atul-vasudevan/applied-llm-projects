from typing import TypedDict, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Local Ollama server (default http://localhost:11434).
llm = ChatOllama(
    model="llama3.1",  # or the model you pulled via `ollama pull`
    temperature=0
)


class State(TypedDict, total=False):
    user_instruction: str
    input_text: str
    task: str
    result: str
    debug: Dict[str, Any]


def _clean_str(s: str) -> str:
    """This helper will be used to clean the router's raw text into a simple keyword."""
    return s.strip().strip('"').strip("'").lower()


def classify_task(state: State) -> State:
    """
    Decide what the user wants:
    - summarize       -> short summary
    - bullet_points   -> bulleted list
    - sentiment       -> sentiment analysis

    Conceptually:
    - We prompt the LLM: "You're a router, output ONE word only."
    - Then we normalise that into one of three options.
    """
    system = SystemMessage(
        content=(
            "You are a router. Based on the user instruction and the text, "
            "choose exactly one of: summarize, bullet_points, sentiment. "
            "Return ONLY that single word."
        )
    )
    human = HumanMessage(
        content=f"Instruction: {state['user_instruction']}\n\nText:\n{state['input_text']}"
    )

    resp = llm.invoke([system, human])
    choice = _clean_str(resp.content)

    if "bullet" in choice:
        task = "bullet_points"
    elif "sentiment" in choice:
        task = "sentiment"
    else:
        task = "summarize"

    debug = dict(state.get("debug", {}))
    debug["router_raw"] = resp.content

    # We return only the fields we want to update.
    return {
        "task": task,
        "debug": debug,
    }


def summarize_text(state: State) -> State:
    """
    Take `input_text` and summarise it into 3–5 sentences.
    """
    system = SystemMessage(
        content="You write clear, concise summaries in 3–5 sentences."
    )
    human = HumanMessage(
        content=f"Summarize the following text:\n\n{state['input_text']}"
    )
    resp = llm.invoke([system, human])

    return {
        "result": resp.content,
    }


def bullet_points(state: State) -> State:
    """
    Take `input_text` and extract key points as bullets.
    """
    system = SystemMessage(
        content="You extract key points as a concise bullet list."
    )
    human = HumanMessage(
        content=f"Extract the key points as bullets from this text:\n\n{state['input_text']}"
    )
    resp = llm.invoke([system, human])

    return {
        "result": resp.content,
    }


def analyze_sentiment(state: State) -> State:
    """
    Perform a simple sentiment analysis: positive/negative/neutral + justification.
    """
    system = SystemMessage(
        content=(
            "You are a sentiment analyst. "
            "Describe sentiment (positive/negative/neutral) and briefly justify it."
        )
    )
    human = HumanMessage(
        content=f"Analyze the sentiment of this text:\n\n{state['input_text']}"
    )
    resp = llm.invoke([system, human])

    return {
        "result": resp.content,
    }


def finalize_answer(state: State) -> State:
    """
    Wrap the result in a small header showing which task was chosen.
    """
    task = state.get("task", "summarize")
    header = f"[Task: {task}]\n"
    body = state.get("result", "")

    return {
        "result": header + body
    }
