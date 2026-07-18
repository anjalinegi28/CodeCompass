from app.ingestion.loader import load_folder


def test_load_folder_walks_entire_project_recursively(tmp_path):
    # Build a small fake "project" with nested folders, mixed file types,
    # and a directory that should be excluded (node_modules).
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "node_modules" / "some_pkg").mkdir(parents=True)

    (tmp_path / "src" / "auth" / "login.py").write_text("def login():\n    pass\n")
    (tmp_path / "docs" / "README.md").write_text("# Docs\n")
    (tmp_path / "node_modules" / "some_pkg" / "index.js").write_text("module.exports = {}\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # should be skipped, not a text ext

    loaded = load_folder(tmp_path)
    paths = {f.path for f in loaded}

    assert "src/auth/login.py" in paths
    assert "docs/README.md" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any(p.endswith(".png") for p in paths)


def test_load_folder_raises_for_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist"
    try:
        load_folder(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
