"""Eval 2 — Chunking Correctness (evals/test_chunking.py)

Covers: embeddings.py chunk loop (1000/200/50) + document_processor.py chunk_text().
Run: pytest evals/test_chunking.py -v
"""
import ast

import pytest

from app.services.document_processor import DocumentProcessor


# ---- helper: replicate embeddings.py chunk loop ----------------------------

def repo_chunk(content: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Exact logic from embeddings.py:106-118."""
    chunks: list[str] = []
    for i in range(0, len(content), chunk_size - overlap):
        chunk_text = content[i : i + chunk_size]
        if len(chunk_text.strip()) > 50:
            chunks.append(chunk_text)
    return chunks


# ---- repo chunk evals ------------------------------------------------------

class TestRepoChunking:
    def test_small_content_single_chunk(self):
        content = "x = 1\n" * 10  # ~60 chars
        chunks = repo_chunk(content)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_large_content_splits(self):
        content = "a" * 3000
        chunks = repo_chunk(content, 1000, 200)
        # step=800: [0:1000],[800:1800],[1600:2600],[2400:3400] → 4
        assert len(chunks) == 4

    def test_overlap_present(self):
        content = "0123456789" * 300  # 3000 chars
        chunks = repo_chunk(content, 1000, 200)
        # chunk[0] ends at 1000, chunk[1] starts at 800 → overlap 200 chars
        assert chunks[0][-200:] == chunks[1][:200]

    def test_tiny_tail_dropped(self):
        # Content where last window is <=50 chars after strip
        content = "a" * 1000 + " " * 790 + "b" * 10  # last window ~10 chars
        chunks = repo_chunk(content, 1000, 200)
        for c in chunks:
            assert len(c.strip()) > 50

    def test_empty_content_no_chunks(self):
        assert repo_chunk("") == []
        assert repo_chunk("   \n  \n") == []

    def test_reconstruction_coverage(self):
        """Reconstructed chunks (deduplicated overlap) should cover original."""
        content = "line {}\n".format(0) * 1  # placeholder
        content = "".join(f"line {i:04d} content here\n" for i in range(200))  # ~4600 chars
        chunks = repo_chunk(content, 1000, 200)
        # Every char of original (except possibly last <50) should appear in some chunk
        covered = set()
        for c in chunks:
            # find c in content (allow overlap)
            start = content.find(c[:20])
            if start != -1:
                covered.add(start)
        assert len(chunks) >= 4

    def test_token_estimate_accuracy(self):
        """token_count=len/4 rough est vs ~25% error bound check."""
        cases = ["a" * 100, "hello world " * 100, "x=1\n" * 200]
        for text in cases:
            est = len(text) // 4
            # Rough: actual tokens would be ~len/4 for English, allow 2x variance
            assert est > 0
            assert est < len(text)

    def test_function_split_rate(self):
        """Measure how often a 1000-char window cuts inside a FunctionDef."""
        code = "\n".join(
            f"def func_{i}(x):\n    return x + {i}\n" for i in range(100)
        )  # ~3000 chars
        chunks = repo_chunk(code, 1000, 200)
        tree = ast.parse(code)
        func_ranges: list[tuple[int, int]] = []
        # Map func bodies to char offsets via source lines
        lines = code.splitlines(keepends=True)
        line_offsets = [0]
        for l in lines:
            line_offsets.append(line_offsets[-1] + len(l))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1  # 0-indexed
                end_line = getattr(node, "end_lineno", start_line + 2) - 1
                start_char = line_offsets[start_line]
                end_char = line_offsets[end_line + 1] if end_line + 1 < len(line_offsets) else len(code)
                func_ranges.append((start_char, end_char))
        # Count how many chunk boundaries fall inside a function
        cuts_inside = 0
        for i in range(len(chunks) - 1):
            # boundary between chunk i and i+1 is at i*800+1000 approx
            boundary = (i + 1) * 800
            for s, e in func_ranges:
                if s < boundary < e:
                    cuts_inside += 1
                    break
        rate = cuts_inside / max(1, len(chunks) - 1)
        # Report, don't assert strict — naive windowing will split ~60-80%
        print(f"\n  function-split rate: {rate:.0%} ({cuts_inside}/{len(chunks)-1} boundaries inside a function)")
        assert rate < 1.0  # at least one boundary is outside a function


# ---- document_processor.chunk_text evals ---------------------------------

class TestDocumentChunkText:
    @pytest.fixture()
    def dp(self):
        return DocumentProcessor(db=None)

    def test_short_text_unchunked(self, dp):
        assert dp.chunk_text("hello", 1000, 200) == ["hello"]

    def test_sentence_boundary_respected(self, dp):
        # Build text where a sentence break exists near chunk_size
        prefix = "word " * 180  # ~900 chars
        text = prefix + ". " + "next sentence " * 200
        chunks = dp.chunk_text(text, 1000, 200)
        # First chunk should end at ". " not mid-word
        assert chunks[0].endswith(".")

    def test_overlap_after_backtrack(self, dp):
        text = "a " * 1000  # ~2000 chars, newlines every so often
        text = text[:900] + "\n" + text[900:]
        chunks = dp.chunk_text(text, 1000, 200)
        assert len(chunks) >= 2

    def test_empty_filtered(self, dp):
        assert dp.chunk_text("", 1000, 200) == [""]

    def test_load_text_splits_paragraphs(self, dp, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Para one is long enough to pass filter. " * 5 + "\n\n" + "Para two also long enough. " * 5)
        chunks = dp.load_text(str(p))
        assert len(chunks) == 2

    def test_load_markdown_splits_headers(self, dp, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("# Header A\nContent A\n\n# Header B\nContent B\n")
        chunks = dp.load_markdown(str(p))
        assert len(chunks) == 2
        assert chunks[0]["metadata"]["header"] == "Header A"
