"""
Stale-doc watcher: a small LangGraph agent that compares a doc's embedding
against a fresh summary-embedding of the code it documents, and flags the
doc as stale if semantic drift is high. Optionally drafts an updated doc.

Graph shape:

    summarize_code -> embed_both -> compare -> (draft_update if stale) -> END

This runs as part of CI on every PR alongside the RAGAS eval gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from app.ingestion.embedder import embed_text
from app.rag.providers import complete

STALE_THRESHOLD = 0.55  # cosine similarity below this => flagged stale

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize what this code does in 2-3 sentences, focused on behavior "
    "a developer would need to know to keep documentation accurate."
)

DRAFT_SYSTEM_PROMPT = (
    "You update stale documentation. Given the old doc and a summary of what "
    "the code now does, rewrite the doc so it accurately reflects the current "
    "code. Keep the same style/format as the original. Output only the "
    "updated doc text."
)


class WatcherState(TypedDict, total=False):
    doc_path: str
    doc_text: str
    code_text: str
    code_summary: str
    doc_embedding: list[float]
    code_embedding: list[float]
    similarity: float
    is_stale: bool
    drafted_update: str | None


@dataclass
class StaleDocResult:
    doc_path: str
    similarity: float
    is_stale: bool
    drafted_update: str | None


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _node_summarize_code(state: WatcherState) -> WatcherState:
    result = complete(SUMMARIZE_SYSTEM_PROMPT, state["code_text"])
    state["code_summary"] = result.text
    return state


def _node_embed_both(state: WatcherState) -> WatcherState:
    state["doc_embedding"] = embed_text(state["doc_text"])
    state["code_embedding"] = embed_text(state["code_summary"])
    return state


def _node_compare(state: WatcherState) -> WatcherState:
    similarity = _cosine(state["doc_embedding"], state["code_embedding"])
    state["similarity"] = similarity
    state["is_stale"] = similarity < STALE_THRESHOLD
    return state


def _node_draft_update(state: WatcherState) -> WatcherState:
    if not state["is_stale"]:
        state["drafted_update"] = None
        return state
    prompt = (
        f"Old doc:\n{state['doc_text']}\n\n"
        f"Current code summary:\n{state['code_summary']}"
    )
    result = complete(DRAFT_SYSTEM_PROMPT, prompt)
    state["drafted_update"] = result.text
    return state


def _build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(WatcherState)
    graph.add_node("summarize_code", _node_summarize_code)
    graph.add_node("embed_both", _node_embed_both)
    graph.add_node("compare", _node_compare)
    graph.add_node("draft_update", _node_draft_update)

    graph.set_entry_point("summarize_code")
    graph.add_edge("summarize_code", "embed_both")
    graph.add_edge("embed_both", "compare")
    graph.add_edge("compare", "draft_update")
    graph.add_edge("draft_update", END)

    return graph.compile()


def check_doc_staleness(doc_path: str, doc_text: str, code_text: str, draft_fix: bool = True) -> StaleDocResult:
    """Run the watcher graph for one (doc, code) pair."""
    app_graph = _build_graph()
    initial: WatcherState = {"doc_path": doc_path, "doc_text": doc_text, "code_text": code_text}
    final_state = app_graph.invoke(initial)

    return StaleDocResult(
        doc_path=doc_path,
        similarity=final_state["similarity"],
        is_stale=final_state["is_stale"],
        drafted_update=final_state.get("drafted_update") if draft_fix else None,
    )
