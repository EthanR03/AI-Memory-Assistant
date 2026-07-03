"""Stage 4 - Store: save vectors + text + metadata in ChromaDB.

ChromaDB is a vector database. It keeps each embedding alongside the
original text and any metadata, and can later find the chunks whose
vectors are closest to a query vector. `PersistentClient` writes all of
this to disk, so your memories survive between runs.
"""
from pathlib import Path

import chromadb


def get_collection(chroma_dir: Path, name: str):
    """Open (or create) a persistent collection stored at `chroma_dir`."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(name=name)


def add_chunks(collection, ids, embeddings, documents, metadatas) -> None:
    """Insert or update chunks in the collection.

    We use `upsert` rather than `add` so that re-running ingestion on the
    same files *updates* existing entries instead of erroring on duplicate
    ids. That makes the pipeline safe to run as many times as you like.
    """
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
