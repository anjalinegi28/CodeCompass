"""
Chunker: splits each loaded file into overlapping text chunks, tracking the
starting line number of each chunk so the RAG agent can cite "file:line"
later instead of just "file".
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.ingestion.loader import LoadedFile


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    text: str


def _chunk_lines(lines: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    """
    Groups lines into character-budgeted windows with overlap.
    Returns (start_line, end_line, text) 1-indexed.
    """
    windows: list[tuple[int, int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        buf: list[str] = []
        char_count = 0
        start = i
        while i < n and char_count < chunk_size:
            buf.append(lines[i])
            char_count += len(lines[i]) + 1
            i += 1
        end = i
        windows.append((start + 1, end, "\n".join(buf)))

        if end >= n:
            break

        # step back for overlap, measured in lines (approximate via char budget)
        back_chars = 0
        step_back = 0
        while step_back < end - start and back_chars < overlap:
            step_back += 1
            back_chars += len(lines[end - step_back]) + 1
        i = max(start + 1, end - step_back)

    return windows


def chunk_file(loaded: LoadedFile, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    lines = loaded.content.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    for idx, (start, end, text) in enumerate(_chunk_lines(lines, chunk_size, overlap)):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{loaded.path}::{idx}",
                file_path=loaded.path,
                start_line=start,
                end_line=end,
                text=text,
            )
        )
    return chunks


def chunk_files(loaded_files: list[LoadedFile]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for f in loaded_files:
        all_chunks.extend(chunk_file(f))
    return all_chunks
