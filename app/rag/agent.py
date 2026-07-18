"""
The RAG agent: retrieves the most relevant chunks for a question, asks the
configured LLM to answer using only that context, and returns the answer
along with the exact file/line citation(s) it was grounded in.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.rag.providers import complete
from app.vectorstore import get_store
from app.vectorstore.base import SearchResult

SYSTEM_PROMPT = """You are CodeCompass, an assistant that answers questions about a codebase.
Rules:
- Answer ONLY using the provided context chunks. If the context doesn't contain
  the answer, say you don't have enough information — do not guess.
- Be concise and specific (2-5 sentences unless asked for more detail).
- Do not invent file paths or line numbers; the citation is added separately
  from the retrieved metadata, not by you.
"""


@dataclass
class Answer:
    text: str
    citations: list[str]  # e.g. ["auth/login.py:40-46"]
    sources: list[SearchResult]
    provider: str
    model: str


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"### {r.file_path} (lines {r.start_line}-{r.end_line})\n```\n{r.text}\n```"
        )
    return "\n\n".join(blocks)


def ask(question: str, k: int = 5, provider: str | None = None) -> Answer:
    store = get_store()
    results = store.query(question, k=k)

    if not results:
        return Answer(
            text="I don't have any indexed content to answer from yet — run ingestion first.",
            citations=[],
            sources=[],
            provider=provider or "none",
            model="none",
        )

    context = _format_context(results)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    result = complete(SYSTEM_PROMPT, user_prompt, provider=provider)

    citations = [f"{r.file_path}:{r.start_line}-{r.end_line}" for r in results]

    return Answer(
        text=result.text,
        citations=citations,
        sources=results,
        provider=result.provider,
        model=result.model,
    )
