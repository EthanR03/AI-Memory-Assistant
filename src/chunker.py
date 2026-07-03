"""Stage 2 - Chunk: split long text into smaller, overlapping pieces.

Why chunk? Embedding models have size limits, and retrieval works far
better when each piece is focused - you want to pull back the relevant
paragraph later, not a whole document.

Why overlap? If an idea straddles a chunk boundary, the overlap ensures
at least one chunk still contains the whole idea, so search doesn't miss it.
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into chunks of about `chunk_size` characters.

    Chunks try to end on a space so we never cut a word in half, and each
    chunk shares `overlap` characters with the one before it.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # Prefer to break on a space, but only look in the part of the
        # window past the overlap zone, so we always move forward.
        if end < n:
            last_space = text.rfind(" ", start + overlap, end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        # Step forward, leaving `overlap` characters behind us. The guard
        # makes forward progress impossible to stall, even on odd input.
        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks
