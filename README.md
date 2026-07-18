# CodeCompass

Agentic RAG chatbot for codebases, with eval-gated CI/CD.

CodeCompass answers questions about a codebase ("how does X work?"), citing the
exact file and line the answer came from. Every pull request automatically
gets its answer quality scored (faithfulness, context precision, context
recall) using RAGAS. If the score drops below a threshold, the PR is blocked
from merging — just like a normal failing test, except this test measures
answer quality instead of code correctness.

## Why "folder upload" is a first-class feature here

Most AI chat UIs (ChatGPT, Claude.ai, etc.) only let you attach individual
files, because a browser file picker can't easily upload an entire directory
tree with structure preserved. CodeCompass sidesteps this entirely: it never
relies on a chat window's uploader. Instead you point it at a folder on disk
(or a zipped folder), and it does its own recursive directory walk — reading
every file, preserving relative paths, and skipping binaries/junk — through:

- the CLI: `python cli.py ingest /path/to/your/project`
- the API: `POST /ingest` with a `folder_path` on the server, OR a `.zip` file
  upload that the server unpacks and walks itself

This means you give it a whole repo, not one file at a time.

## Project layout

```
codecompass/
├── app/
│   ├── config.py            # settings, env vars, provider keys
│   ├── ingestion/
│   │   ├── loader.py        # recursive folder walker (accepts folder OR zip)
│   │   ├── chunker.py       # splits files into overlapping text chunks
│   │   └── embedder.py      # Sentence-Transformers embeddings
│   ├── vectorstore/
│   │   ├── chroma_store.py  # ChromaDB backend
│   │   └── faiss_store.py   # FAISS backend (swappable)
│   ├── rag/
│   │   ├── providers.py     # switch between OpenAI / Gemini / Claude
│   │   └── agent.py         # LangChain RAG chain, returns answer + citation
│   ├── stale_docs/
│   │   └── watcher.py       # LangGraph agent: flags docs stale vs new code
│   ├── eval/
│   │   ├── testset.json     # sample Q&A eval set
│   │   └── ragas_eval.py    # scores faithfulness/precision/recall, exits non-zero on fail
│   ├── mlflow_logging/
│   │   └── logger.py        # logs every eval run's params + scores to MLflow
│   └── api/
│       └── main.py          # FastAPI app: /ingest /ask /eval /stale-docs
├── cli.py                   # local command-line interface
├── tests/                   # pytest unit tests
├── .github/workflows/eval-gate.yml   # CI/CD: blocks merge if RAGAS scores dip
├── Dockerfile
├── docker-compose.yml        # app + chromadb + mlflow, one command
├── requirements.txt
└── .env.example
```

## Quick start

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key(s)

# 2. Ingest an entire project folder (this is the "give it the whole project" step)
python cli.py ingest /path/to/some/repo

# 3. Ask a question
python cli.py ask "How does the authentication middleware work?"

# 4. Run the eval suite manually (same thing CI runs on every PR)
python -m app.eval.ragas_eval

# 5. Or run everything with Docker (app + vector DB + MLflow UI)
docker compose up --build
```

## How the CI/CD gate works

`.github/workflows/eval-gate.yml` runs on every pull request:

1. Re-ingests the repo at the PR's commit
2. Runs the fixed eval question set through the RAG agent
3. Scores every answer with RAGAS (faithfulness, context precision, context recall)
4. Logs the run to MLflow (params: chunk size, embedding model, LLM provider; results: scores)
5. Fails the workflow (blocking merge, if branch protection requires this check) if any score
   drops below the threshold set in `app/config.py`

## Swapping LLM providers

Set `LLM_PROVIDER=openai|gemini|claude` in `.env`. `app/rag/providers.py` is a
thin factory — add a new provider by adding one function there.

## Notes on scope

This is a full working scaffold: ingestion, chunking, embedding, vector
storage, RAG querying with citations, stale-doc detection, RAGAS evaluation,
MLflow logging, a FastAPI service, a CLI, Docker packaging, and a real GitHub
Actions gate. To actually run end-to-end you'll need to supply your own LLM
API key(s) in `.env` — everything else runs locally out of the box.
