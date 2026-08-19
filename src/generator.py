"""Milestone 3 - Generate: answer a question using retrieved chunks as context.

This is the "G" in RAG. The chat model does not remember anything by
itself - on every question we hand it the retrieved chunks as context
and instruct it to answer ONLY from them. That grounding is what keeps
answers tied to your actual notes instead of the model's imagination.
"""
from .embedder import _get_client  # reuse the same OpenAI client
from .retriever import retrieve

# A small, cheap, fast chat model - plenty for answering from context.
CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a personal memory assistant. Answer the user's question using ONLY
the context provided below. Each context block is labeled with its source
file.

Rules:
- If the context contains the answer, give it concisely and mention which
  source file it came from.
- If the context does NOT contain the answer, say you don't have a memory
  about that. Do not guess or use outside knowledge.
"""


def _format_context(hits: list[dict]) -> str:
    """Turn retrieved chunks into labeled blocks the model can cite."""
    blocks = []
    for hit in hits:
        blocks.append(
            f"[source: {hit['source']}, chunk {hit['chunk_index']}]\n"
            f"{hit['text']}"
        )
    return "\n\n".join(blocks)


def answer(question: str, n_results: int = 4) -> dict:
    """Retrieve relevant chunks, then generate a grounded answer.

    Returns {"answer": str, "sources": list of the chunks used} so the
    caller can show where the answer came from.
    """
    # Step 1 - Retrieve (Milestone 2 doing its job)
    hits = retrieve(question, n_results=n_results)

    # Step 2 - Generate
    context = _format_context(hits) if hits else "(no stored memories)"
    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": hits,
    }
