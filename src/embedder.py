"""Stage 3 - Embed: turn chunks of text into vectors using OpenAI.

An embedding is a list of numbers that captures the *meaning* of a piece
of text. Similar meaning -> similar numbers. That is what lets us later
search by meaning instead of by keyword.
"""
from openai import OpenAI

# OpenAI caps how many texts you can embed in one request, so we send them
# in batches. Smaller data will fit in a single batch anyway.
BATCH_SIZE = 100

# We create the client lazily (on first use) so that the .env file is
# guaranteed to be loaded by the time we need the API key.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    """Return one embedding vector per input string, in the same order.

    Sending many texts per request is much faster and cheaper than one
    request per chunk.
    """
    if not texts:
        return []

    client = _get_client()
    vectors: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        # response.data is index-tagged; sort so order matches our input.
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)

    return vectors
