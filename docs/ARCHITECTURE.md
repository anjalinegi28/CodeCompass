# Architecture

```
Whole project folder (or .zip)
        |
        v
  loader.py          -- recursive walk, skips node_modules/.git/venv/etc.
        |
        v
  chunker.py          -- overlapping windows, tracks start/end line per chunk
        |
        v
  embedder.py          -- Sentence-Transformers -> vectors
        |
        v
  vectorstore/         -- ChromaDB (default) or FAISS, same interface
        |
        v
  rag/agent.py          -- retrieves top-k chunks, asks configured LLM,
        |                  returns answer + file:line citations
        v
  eval/ragas_eval.py     -- runs fixed question set through the agent,
        |                  scores with RAGAS, logs to MLflow
        v
  GitHub Actions gate     -- fails the PR check if any score < threshold
```

## Why folders, not files

`app/ingestion/loader.py` has three entry points:

- `load_folder(path)` — walks a real directory recursively
- `load_zip(path)` — unpacks a zipped folder, then reuses `load_folder`
- `load_paths([...])` — still supports handing over individual files

The CLI and the `/ingest/folder` API route call `load_folder` directly. The
`/ingest/zip` API route exists for the case where an actual folder path
on the server isn't available to the caller (e.g. calling the API from a
browser) — a zip is the one artifact a normal upload button can carry with
directory structure intact, and the server unpacks it before walking it the
same way.

## Why the eval gate matters

Changing chunk size, swapping embedding models, or switching the vector
store backend can silently make retrieval worse, which makes answers worse,
which nobody notices until a user complains. `ragas_eval.py` turns "does the
chatbot still work" into an automated, versioned, blocking check — the same
category of protection normal unit tests give application code, applied to
answer quality instead.
