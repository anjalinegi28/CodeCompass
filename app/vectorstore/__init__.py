"""
Vector store abstraction. `get_store(collection_name)` returns whichever
backend is configured (chroma or faiss) behind one shared interface:

    store.add(chunks)
    store.query(text, k=5) -> list[SearchResult]

`collection_name` lets callers keep separate visitors'/sessions' data in
isolated collections (used by the public multi-user API) instead of one
shared collection. Defaults to a single shared collection for local/CLI use.
"""
from __future__ import annotations

from app.config import settings

DEFAULT_COLLECTION_NAME = "codecompass_chunks"


def get_store(collection_name: str = DEFAULT_COLLECTION_NAME):
    if settings.vector_store == "faiss":
        from app.vectorstore.faiss_store import FaissStore

        return FaissStore(collection_name=collection_name)
    from app.vectorstore.chroma_store import ChromaStore

    return ChromaStore(collection_name=collection_name)
