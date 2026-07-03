"""Milestone 1 - the ingestion pipeline.

Run it from your project root with:

    python -m src.ingest

It reads every .txt / .md file in `data/`, splits each into overlapping
chunks, embeds the chunks with OpenAI, and stores them in a local
ChromaDB collection. Safe to run repeatedly - existing chunks are updated
in place rather than duplicated.
"""
import hashlib
from datetime import datetime, timezone

from . import config
from .loader import load_documents
from .chunker import chunk_text
from .embedder import embed_texts
from .store import get_collection, add_chunks


def _make_id(source: str, index: int) -> str:
    """A stable, unique id for one chunk.

    It is derived from the file name plus the chunk's position, so
    re-ingesting the same file overwrites its chunks instead of piling up
    duplicates.
    """
    raw = f"{source}:{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def run() -> None:
    # Make sure the folders we depend on exist.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1 - Load
    documents = load_documents(config.DATA_DIR)
    if not documents:
        print(f"No .txt or .md files found in {config.DATA_DIR}.")
        print("Drop a file in there and run this again.")
        return
    print(f"Loaded {len(documents)} file(s).")

    collection = get_collection(config.CHROMA_DIR, config.COLLECTION_NAME)
    ingested_at = datetime.now(timezone.utc).isoformat()
    total_chunks = 0

    # Process one file at a time so progress is easy to follow.
    for doc in documents:
        source = doc["source"]

        # Stage 2 - Chunk
        chunks = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            continue

        # Stage 3 - Embed
        embeddings = embed_texts(chunks, config.EMBEDDING_MODEL)

        # Stage 4 - Store
        ids = [_make_id(source, i) for i in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": i, "ingested_at": ingested_at}
            for i in range(len(chunks))
        ]
        add_chunks(collection, ids, embeddings, chunks, metadatas)

        total_chunks += len(chunks)
        print(f"  {source}: {len(chunks)} chunk(s) embedded and stored.")

    print(
        f"\nDone. {total_chunks} chunk(s) now in the "
        f"'{config.COLLECTION_NAME}' collection."
    )
    print(f"Vector store location: {config.CHROMA_DIR}")


if __name__ == "__main__":
    run()
