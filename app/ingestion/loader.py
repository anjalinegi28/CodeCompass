"""
Loader: turns "a whole project" into a list of (relative_path, text) pairs.

This is the piece that makes CodeCompass accept entire project folders
instead of one file at a time like a typical chat UI upload box:

- load_folder(path)   -> walks a directory recursively on disk
- load_zip(zip_path)  -> unpacks a zipped project folder, then walks it
- load_paths(paths)   -> loads an explicit list of individual files (still
                         supported, for when someone really does just want
                         to hand over a handful of files)

All three return the same shape so the rest of the pipeline (chunker,
embedder) doesn't need to know or care how the files arrived.
"""
from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import settings


@dataclass
class LoadedFile:
    path: str          # path relative to the project root, e.g. "src/auth/login.py"
    content: str        # raw text content
    size_bytes: int


def _is_excluded_dir(dir_name: str) -> bool:
    return dir_name in settings.excluded_dirs or dir_name.endswith(".egg-info")


def _should_ingest(file_path: Path) -> bool:
    if file_path.suffix.lower() not in settings.allowed_extensions:
        return False
    try:
        if file_path.stat().st_size > settings.max_file_size_bytes:
            return False
    except OSError:
        return False
    return True


def load_folder(root: str | Path) -> list[LoadedFile]:
    """
    Recursively walk an entire project folder on disk and return every
    ingestible file inside it, with paths relative to `root`.

    This is the core of the "give it the whole project" feature — point it
    at a repo root and it reads everything, preserving the folder structure
    in each file's relative path (used later for citations like
    'auth/login.py:42').
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    results: list[LoadedFile] = []

    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(_is_excluded_dir(part) for part in path.relative_to(root).parts[:-1]):
            continue
        if not _should_ingest(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Always use forward slashes for the relative path, even on Windows,
        # so citations like "src/auth/login.py:40-46" look the same on every OS.
        rel = path.relative_to(root).as_posix()
        results.append(LoadedFile(path=rel, content=text, size_bytes=path.stat().st_size))

    return results


def load_zip(zip_path: str | Path) -> list[LoadedFile]:
    """
    Accepts a zipped project folder (e.g. uploaded through the API as a
    single .zip, since that's the one thing a browser upload button *can*
    carry with folder structure intact). Unpacks to a temp dir, then reuses
    load_folder so behavior is identical either way.
    """
    zip_path = Path(zip_path)
    with tempfile.TemporaryDirectory(prefix="codecompass_upload_") as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        return load_folder(tmp)


def load_paths(paths: Iterable[str | Path]) -> list[LoadedFile]:
    """Load a specific list of individual files (single-file mode, still supported)."""
    results: list[LoadedFile] = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists() or not path.is_file():
            continue
        if not _should_ingest(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        results.append(LoadedFile(path=path.name, content=text, size_bytes=path.stat().st_size))
    return results
