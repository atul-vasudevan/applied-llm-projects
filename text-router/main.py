from dotenv import load_dotenv
load_dotenv()

from src.graph import graph
from src.nodes import State


def run_once():
    print("== Text Task Router Demo ==")
    user_instruction = input(
        "What would you like me to do? (e.g. summarize / bullets / sentiment)\n> "
    )
    print("\nPaste your text (end with an empty line):")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    input_text = "\n".join(lines)

    initial_state: State = {
        "user_instruction": user_instruction,
        "input_text": input_text,
        "task": "",
        "result": "",
        "debug": {},
    }

    config = {
        "configurable": {
            "run_name": "text_task_router_run",
            "user_id": "demo-user",
        }
    }

    final_state = graph.invoke(initial_state, config=config)

    print("\n=== ROUTED TASK ===")
    print(final_state.get("task"))
    print("\n=== RESULT ===")
    print(final_state.get("result"))


if __name__ == "__main__":
    run_once()