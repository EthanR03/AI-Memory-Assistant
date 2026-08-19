"""Ask your memory assistant a question from the command line.

Run from the project root:

    python -m src.ask "What is the goal of the project?"

Or run with no arguments for an interactive session.
"""
import sys

from .generator import answer


def _print_answer(result: dict) -> None:
    print("\n" + result["answer"])
    if result["sources"]:
        used = {hit["source"] for hit in result["sources"]}
        print(f"\n(retrieved from: {', '.join(sorted(used))})")


def run() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _print_answer(answer(question))
        return

    print("Memory assistant. Ask a question, or 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        _print_answer(answer(question))


if __name__ == "__main__":
    run()
