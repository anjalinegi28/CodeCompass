"""
Vector store abstraction. `get_store()` returns whichever backend is
configured (chroma or faiss) behind one shared interface:

    store.add(chunks)
    store.query(text, k=5) -> list[SearchResult]
"""
from __future__ import annotations

from app.config import settings


def get_store():
    if settings.vector_store == "faiss":
        from app.vectorstore.faiss_store import FaissStore

        return FaissStore()
    from app.vectorstore.chroma_store import ChromaStore

    return ChromaStore()
