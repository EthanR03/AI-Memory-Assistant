"""Search the memory store from the command line.

Run from the project root:

    python -m src.search "What is the goal of the project?"

Or run it with no arguments to get an interactive prompt.
"""
import sys

from .retriever import retrieve


def _print_hits(hits: list[dict]) -> None:
    if not hits:
        print("No results. Is the store empty? Run: python -m src.ingest")
        return
    for rank, hit in enumerate(hits, start=1):
        print(f"\n--- Result {rank} "
              f"(source: {hit['source']}, chunk {hit['chunk_index']}, "
              f"distance: {hit['distance']:.4f}) ---")
        print(hit["text"])


def run() -> None:
    # Question can come from the command line...
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        _print_hits(retrieve(query))
        return

    # ...or from an interactive loop.
    print("Memory search. Type a question, or 'quit' to exit.")
    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        _print_hits(retrieve(query))


if __name__ == "__main__":
    run()
