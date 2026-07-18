"""
Embedder: turns chunk text into vectors using Sentence-Transformers.

Wrapped in a thin class so the vector store code doesn't need to know
which embedding model or library is behind it.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily so `python cli.py --help` etc. don't pay the import
    # cost / require torch installed just to print usage.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
