"""Milestone 2 - Retrieve: find the stored chunks closest in meaning to a query.

This is the mirror image of ingestion. We embed the *question* with the
same model we used for the chunks, then ask ChromaDB for the nearest
stored vectors. Nearest vectors = nearest meaning.
"""
from . import config
from .embedder import embed_texts
from .store import get_collection


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """Return the `n_results` chunks most relevant to `query`.

    Each result is a dict:
        {
            "text":     the chunk's original text,
            "source":   which file it came from,
            "chunk_index": its position in that file,
            "distance": how far its vector is from the query vector
                        (smaller = more similar),
        }
    """
    # Embed the query with the SAME model used at ingestion time.
    # Mixing models would put query and chunks in different vector
    # spaces, and the distances would be meaningless.
    query_vector = embed_texts([query], config.EMBEDDING_MODEL)[0]

    collection = get_collection(config.CHROMA_DIR, config.COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
    )

    # Chroma returns parallel lists (one per query); we sent one query,
    # so we read index [0] of each and zip them into friendly dicts.
    hits = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": text,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": distance,
            }
        )
    return hits
