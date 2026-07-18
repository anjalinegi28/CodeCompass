from unittest.mock import MagicMock, patch

from app.rag.providers import CompletionResult
from app.vectorstore.base import SearchResult


@patch("app.rag.agent.complete")
@patch("app.rag.agent.get_store")
def test_ask_returns_answer_with_citations(mock_get_store, mock_complete):
    fake_store = MagicMock()
    fake_store.query.return_value = [
        SearchResult(
            chunk_id="auth/login.py::0",
            file_path="auth/login.py",
            start_line=40,
            end_line=46,
            text="def login(user, password): ...",
            score=0.9,
        )
    ]
    mock_get_store.return_value = fake_store
    mock_complete.return_value = CompletionResult(
        text="It hashes the password and compares it to the stored hash.",
        latency_seconds=0.1,
        provider="openai",
        model="gpt-4o-mini",
    )

    from app.rag.agent import ask

    result = ask("How does login check the password?")

    assert "hashes the password" in result.text
    assert result.citations == ["auth/login.py:40-46"]
    assert result.provider == "openai"


@patch("app.rag.agent.get_store")
def test_ask_handles_empty_index_gracefully(mock_get_store):
    fake_store = MagicMock()
    fake_store.query.return_value = []
    mock_get_store.return_value = fake_store

    from app.rag.agent import ask

    result = ask("Anything?")

    assert "haven't" not in result.text  # just sanity, message is defined in agent.py
    assert result.citations == []
