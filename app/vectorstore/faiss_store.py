"""
FAISS vector store backend — a swappable alternative to Chroma, used to
demonstrate/benchmark how retrieval quality shifts when you change the
vector backend (this is exactly the kind of change the RAGAS eval gate is
meant to catch if it degrades answer quality).
"""
from __future__ import annotations

import json
import os

import numpy as np

from app.config import settings
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import embed_texts
from app.vectorstore.base import SearchResult

METADATA_FILE_SUFFIX = ".meta.json"


class FaissStore:
    def __init__(self):
        import faiss

        self._faiss = faiss
        self.index_path = settings.faiss_index_path
        self.meta_path = self.index_path + METADATA_FILE_SUFFIX
        self.index = None
        self.metadatas: list[dict] = []
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = self._faiss.read_index(self.index_path)
            with open(self.meta_path, "r") as f:
                self.metadatas = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        self._faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadatas, f)

    def reset(self):
        self.index = None
        self.metadatas = []
        for p in (self.index_path, self.meta_path):
            if os.path.exists(p):
                os.remove(p)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = np.array(embed_texts([c.text for c in chunks]), dtype="float32")
        if self.index is None:
            self.index = self._faiss.IndexFlatIP(vectors.shape[1])
        self._faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for c in chunks:
            self.metadatas.append(
                {
                    "chunk_id": c.chunk_id,
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "text": c.text,
                }
            )
        self._save()
        return len(chunks)

    def query(self, text: str, k: int = 5) -> list[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = np.array(embed_texts([text]), dtype="float32")
        self._faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            meta = self.metadatas[idx]
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    file_path=meta["file_path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    text=meta["text"],
                    score=float(score),
                )
            )
        return results

    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0
