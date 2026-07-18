"""
ChromaDB vector store backend — the default. Persists to disk so re-running
queries doesn't require re-ingesting the whole project every time.
"""
from __future__ import annotations

from app.config import settings
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import embed_texts
from app.vectorstore.base import SearchResult

COLLECTION_NAME = "codecompass_chunks"


class ChromaStore:
    def __init__(self):
        import chromadb

        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def reset(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"file_path": c.file_path, "start_line": c.start_line, "end_line": c.end_line}
                for c in chunks
            ],
        )
        return len(chunks)

    def query(self, text: str, k: int = 5) -> list[SearchResult]:
        query_embedding = embed_texts([text])[0]
        raw = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        results: list[SearchResult] = []
        if not raw["ids"] or not raw["ids"][0]:
            return results

        for i, chunk_id in enumerate(raw["ids"][0]):
            meta = raw["metadatas"][0][i]
            distance = raw["distances"][0][i] if raw.get("distances") else 0.0
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    file_path=meta["file_path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    text=raw["documents"][0][i],
                    score=1.0 - distance,  # cosine distance -> similarity-ish score
                )
            )
        return results

    def count(self) -> int:
        return self.collection.count()
