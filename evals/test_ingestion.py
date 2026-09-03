"""Eval 1 — Ingestion Completeness (evals/test_ingestion.py)

Covers: ingestion.py scan_repository, extract_file_metadata, is_binary_file,
        LANGUAGE_MAP, IGNORE_PATTERNS, 1MB cap, REPOSITORY_FILE_LIMIT.

All tests are unit-level (no DB, no git clone, no Ollama) so they run in CI
without external services. Run: pytest evals/test_ingestion.py -v
"""
import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from app.services.ingestion import RepositoryIngestionService


@pytest.fixture()
def svc(db_session=None):
    # db not needed for file-level methods
    return RepositoryIngestionService(db=None)


# ---- language detection ----------------------------------------------------

class TestLanguageDetection:
    def test_known_extensions(self, svc):
        assert svc.detect_language("main.py") == "python"
        assert svc.detect_language("app.ts") == "typescript"
        assert svc.detect_language("Dockerfile") is None  # no dot
        assert svc.detect_language("README.md") == "markdown"

    def test_unknown_returns_none(self, svc):
        assert svc.detect_language("file.xyz") is None

    def test_case_insensitive(self, svc):
        assert svc.detect_language("MAIN.PY") == "python"


# ---- binary detection ------------------------------------------------------

class TestBinaryDetection:
    def test_text_file_not_binary(self, svc, tmp_path):
        p = tmp_path / "code.py"
        p.write_text("print('hello')\n", encoding="utf-8")
        assert svc.is_binary_file(str(p)) is False

    def test_null_byte_is_binary(self, svc, tmp_path):
        p = tmp_path / "bin.dat"
        p.write_bytes(b"hello\x00world")
        assert svc.is_binary_file(str(p)) is True

    def test_missing_file_is_binary(self, svc):
        assert svc.is_binary_file("/nonexistent/file.txt") is True


# ---- should_ignore ---------------------------------------------------------

class TestShouldIgnore:
    def test_git_ignored(self, svc):
        assert svc.should_ignore(".git/config") is True
        assert svc.should_ignore("src/.git/hooks") is True

    def test_node_modules_ignored(self, svc):
        assert svc.should_ignore("node_modules/react/index.js") is True

    def test_glob_patterns(self, svc):
        assert svc.should_ignore("bundle.min.js") is True
        assert svc.should_ignore("app.min.css") is True
        assert svc.should_ignore("debug.log") is True

    def test_normal_file_not_ignored(self, svc):
        assert svc.should_ignore("app/services/planner.py") is False
        assert svc.should_ignore("README.md") is False


# ---- extract_file_metadata -------------------------------------------------

class TestExtractFileMetadata:
    def test_text_file_content_read(self, svc, tmp_path):
        p = tmp_path / "hello.py"
        p.write_text("x = 1\n", encoding="utf-8")
        meta = svc.extract_file_metadata(str(tmp_path), "hello.py")
        assert meta["path"] == "hello.py"
        assert meta["language"] == "python"
        assert meta["content"] == "x = 1\n"
        assert meta["metadata"]["is_binary"] is False
        assert meta["content_hash"] == hashlib.sha256(b"x = 1\n").hexdigest()

    def test_binary_content_none(self, svc, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(b"\x89PNG\x00\x00")
        meta = svc.extract_file_metadata(str(tmp_path), "img.png")
        assert meta["content"] is None
        assert meta["metadata"]["is_binary"] is True

    def test_oversized_file_content_none(self, svc, tmp_path):
        p = tmp_path / "big.txt"
        # Create file >1MB (ingestion.py:228)
        p.write_bytes(b"a" * (1024 * 1024 + 1))
        meta = svc.extract_file_metadata(str(tmp_path), "big.txt")
        assert meta["content"] is None
        assert meta["size_bytes"] == 1024 * 1024 + 1


# ---- scan_repository -------------------------------------------------------

class TestScanRepository:
    def test_scan_filters_ignored_dirs(self, svc, tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x=1")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("y=1")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]")

        files = svc.scan_repository(str(tmp_path))
        paths = {f["path"] for f in files}
        assert any("main.py" in p for p in paths)
        assert not any("node_modules" in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_scan_returns_metadata_list(self, svc, tmp_path):
        (tmp_path / "a.py").write_text("a=1")
        (tmp_path / "b.md").write_text("# hi")
        files = svc.scan_repository(str(tmp_path))
        assert len(files) == 2
        for f in files:
            assert "path" in f and "language" in f and "content" in f

    def test_scan_empty_repo(self, svc, tmp_path):
        files = svc.scan_repository(str(tmp_path))
        assert files == []


# ---- coverage guard --------------------------------------------------------

class TestCoverageGuard:
    """Project-level guard: measures silent drop rate."""

    def test_no_silent_drop_on_small_repo(self, svc, tmp_path):
        # Create 10 small text files
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"# file {i}\nx={i}\n")
        files = svc.scan_repository(str(tmp_path))
        # Count files where content is not None
        with_content = sum(1 for f in files if f["content"] is not None)
        drop_rate = 1 - with_content / len(files) if files else 0
        # For small text files drop_rate should be 0
        assert drop_rate == 0, f"Unexpected drop rate {drop_rate:.0%} — {len(files)-with_content}/{len(files)} files lost"
        print(f"\n  ingestion coverage: {with_content}/{len(files)} files with content (drop {drop_rate:.0%})")
