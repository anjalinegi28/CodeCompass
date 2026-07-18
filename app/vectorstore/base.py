from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    text: str
    score: float
