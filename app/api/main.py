"""
CodeCompass API — public multi-user mode.

Each visitor uploads their own project as a .zip. The server ingests it into
an isolated Chroma collection (named after a server-generated session_id),
so different visitors' projects never mix. The visitor gets the session_id
back and sends it along with every /ask call so they only ever query their
own uploaded project.

Session data is cleaned up automatically after SESSION_TTL_SECONDS of
inactivity to keep disk usage bounded on a public demo.
"""
from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.eval.ragas_eval import run_eval
from app.ingestion.chunker import chunk_files
from app.ingestion.loader import load_folder, load_zip
from app.rag.agent import ask as rag_ask
from app.stale_docs.watcher import check_doc_staleness
from app.vectorstore import get_store
from fastapi.concurrency import run_in_threadpool

app = FastAPI(
    title="CodeCompass API",
    description="Agentic RAG over a codebase, with eval-gated CI/CD.",
    version="0.1.0",
)

# --- Serve the simple frontend (static/index.html) --------------------------
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def homepage():
    return FileResponse(STATIC_DIR / "index.html")


# --- Session tracking (in-memory) -------------------------------------------
# Maps session_id -> last_used_at (unix timestamp). Used to (a) verify a
# session_id sent to /ask is real, and (b) clean up old sessions' Chroma
# collections so a public demo's disk doesn't grow forever.
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours of inactivity
_sessions: dict[str, float] = {}


def _collection_name(session_id: str) -> str:
    return f"session_{session_id}"


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, last_used in _sessions.items() if now - last_used > SESSION_TTL_SECONDS]
    for sid in expired:
        try:
            get_store(_collection_name(sid)).delete()
        except Exception:
            pass
        _sessions.pop(sid, None)


def _touch_session(session_id: str) -> None:
    _sessions[session_id] = time.time()


# --- Basic per-IP rate limiting (protects your disk and your LLM API costs) -
_ASK_LIMIT, _ASK_WINDOW = 6, 60          # 6 questions per 60s per IP
_INGEST_LIMIT, _INGEST_WINDOW = 2, 600   # 2 uploads per 10 minutes per IP
_request_log: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(request: Request, limit: int, window: int, bucket: str) -> None:
    ip = request.client.host if request.client else "unknown"
    key = f"{bucket}:{ip}"
    now = time.time()
    recent = [t for t in _request_log[key] if now - t < window]
    if len(recent) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a bit and try again.")
    recent.append(now)
    _request_log[key] = recent


MAX_UPLOAD_SIZE_BYTES = settings.max_upload_size_bytes  # 1 GB — see app/config.py
# generous since node_modules/.git/venv/dist/build are already excluded by
# app/config.py's excluded_dirs, and only text/code file extensions are read
# (see allowed_extensions), so real repos rarely get close to this even
# though the raw zip/folder might look bigger.


class IngestFolderRequest(BaseModel):
    folder_path: str
    reset: bool = False


class IngestResponse(BaseModel):
    session_id: str
    files_ingested: int
    chunks_indexed: int


class AskRequest(BaseModel):
    session_id: str
    question: str
    k: int = 5
    provider: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    provider: str
    model: str


class StaleDocRequest(BaseModel):
    doc_path: str
    doc_text: str
    code_text: str


class StaleDocResponse(BaseModel):
    doc_path: str
    similarity: float
    is_stale: bool
    drafted_update: str | None


@app.post("/ingest/folder", response_model=IngestResponse)
def ingest_folder(req: IngestFolderRequest, request: Request):
    """
    Ingest a project folder that already lives on the server's disk.
    Mainly useful for local/dev use — public visitors use /ingest/zip instead,
    since they can't give the server a path on YOUR machine.
    """
    _check_rate_limit(request, _INGEST_LIMIT, _INGEST_WINDOW, "ingest")
    _cleanup_expired_sessions()

    try:
        loaded_files = load_folder(req.folder_path)
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not loaded_files:
        raise HTTPException(status_code=400, detail="No ingestible files found in that folder.")

    session_id = uuid.uuid4().hex
    store = get_store(_collection_name(session_id))
    chunks = chunk_files(loaded_files)
    indexed = store.add(chunks)
    _touch_session(session_id)

    return IngestResponse(session_id=session_id, files_ingested=len(loaded_files), chunks_indexed=indexed)


@app.post("/ingest/zip", response_model=IngestResponse)
async def ingest_zip(request: Request, file: UploadFile = File(...)):
    """
    The public upload path: a visitor sends a .zip of their project. The
    server ingests it into a brand-new, isolated session and hands back a
    session_id, which the visitor then uses for every /ask call.
    """
    _check_rate_limit(request, _INGEST_LIMIT, _INGEST_WINDOW, "ingest")
    _cleanup_expired_sessions()

    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file of your project folder.")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Zip file is too large (1 GB limit for this demo).")
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        loaded_files = await run_in_threadpool(load_zip, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not loaded_files:
        raise HTTPException(status_code=400, detail="No ingestible files found in that zip.")

    session_id = uuid.uuid4().hex
    store = get_store(_collection_name(session_id))
    chunks = await run_in_threadpool(chunk_files, loaded_files)
    indexed = await run_in_threadpool(store.add, chunks)
    _touch_session(session_id)

    return IngestResponse(session_id=session_id, files_ingested=len(loaded_files), chunks_indexed=indexed)


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    request: Request,
    files: List[UploadFile] = File(...),
    relative_paths: List[str] = Form(...),
):
    """
    The "pick a folder" upload path — used when a visitor's browser lets
    them select a whole folder (via a webkitdirectory file input) instead
    of zipping it first. The browser sends every file in the folder as a
    separate part, plus each file's relative path (so nested structure like
    src/auth/login.py is preserved for citations). The server writes them
    into a temporary folder mirroring that structure, then ingests it the
    same way /ingest/folder does.
    """
    _check_rate_limit(request, _INGEST_LIMIT, _INGEST_WINDOW, "ingest")
    _cleanup_expired_sessions()

    if len(files) != len(relative_paths):
        raise HTTPException(status_code=400, detail="Mismatched files and paths in upload.")
    if not files:
        raise HTTPException(status_code=400, detail="No files were selected.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="codecompass_upload_"))
    total_size = 0

    try:
        for upload, rel_path in zip(files, relative_paths):
            # Guard against path traversal (e.g. "../../etc/passwd") from a
            # malicious client — keep every file inside tmp_dir.
            safe_rel_path = Path(rel_path.replace("\\", "/")).as_posix().lstrip("/")
            dest_path = (tmp_dir / safe_rel_path).resolve()
            if tmp_dir.resolve() not in dest_path.parents and dest_path != tmp_dir.resolve():
                continue  # skip anything that tries to escape tmp_dir

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            content = await upload.read()
            total_size += len(content)
            if total_size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="Folder is too large (1 GB total limit for this demo).",
                )
            dest_path.write_bytes(content)

        loaded_files = await run_in_threadpool(load_folder, str(tmp_dir))
        if not loaded_files:
            raise HTTPException(status_code=400, detail="No ingestible files found in that folder.")

        session_id = uuid.uuid4().hex
        store = get_store(_collection_name(session_id))
        chunks = await run_in_threadpool(chunk_files, loaded_files)
        indexed = await run_in_threadpool(store.add, chunks)
        _touch_session(session_id)

        return IngestResponse(session_id=session_id, files_ingested=len(loaded_files), chunks_indexed=indexed)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    _check_rate_limit(request, _ASK_LIMIT, _ASK_WINDOW, "ask")

    if req.session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please upload your project again.",
        )
    _touch_session(req.session_id)

    result = rag_ask(req.question, k=req.k, provider=req.provider, collection_name=_collection_name(req.session_id))
    return AskResponse(
        answer=result.text,
        citations=result.citations,
        provider=result.provider,
        model=result.model,
    )


@app.post("/stale-docs", response_model=StaleDocResponse)
def stale_docs(req: StaleDocRequest):
    result = check_doc_staleness(req.doc_path, req.doc_text, req.code_text)
    return StaleDocResponse(
        doc_path=result.doc_path,
        similarity=result.similarity,
        is_stale=result.is_stale,
        drafted_update=result.drafted_update,
    )


@app.post("/eval")
def eval_run():
    """Runs the same RAGAS eval gate that CI runs on every PR, on demand."""
    scores = run_eval()
    return {"scores": scores}


@app.get("/health")
def health():
    return {"status": "ok"}