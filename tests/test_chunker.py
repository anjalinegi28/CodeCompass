from app.ingestion.chunker import chunk_file
from app.ingestion.loader import LoadedFile


def test_chunk_file_produces_at_least_one_chunk():
    content = "\n".join(f"line {i}" for i in range(50))
    loaded = LoadedFile(path="sample.py", content=content, size_bytes=len(content))

    chunks = chunk_file(loaded, chunk_size=200, overlap=20)

    assert len(chunks) > 0
    assert all(c.file_path == "sample.py" for c in chunks)
    assert all(c.start_line >= 1 for c in chunks)
    assert all(c.end_line >= c.start_line for c in chunks)


def test_chunk_file_empty_content_returns_no_chunks():
    loaded = LoadedFile(path="empty.py", content="", size_bytes=0)
    chunks = chunk_file(loaded)
    assert chunks == []


def test_chunks_cover_the_whole_file_with_overlap():
    lines = [f"line {i}" for i in range(30)]
    content = "\n".join(lines)
    loaded = LoadedFile(path="sample.py", content=content, size_bytes=len(content))

    chunks = chunk_file(loaded, chunk_size=100, overlap=15)

    # Last chunk should reach (or very nearly reach) the end of the file.
    assert chunks[-1].end_line == len(lines)
