"""Tests for llmfs.compression.chunker"""
import pytest
from llmfs.compression.chunker import AdaptiveChunker, Chunk


@pytest.fixture
def chunker():
    return AdaptiveChunker(code_target=50, prose_target=30, plain_target=30, overlap=5)


class TestAdaptiveChunker:
    def test_empty_returns_one_chunk(self, chunker):
        chunks = chunker.chunk("   ")
        assert len(chunks) == 1

    def test_short_content_single_chunk(self, chunker):
        chunks = chunker.chunk("hello world")
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "hello world" in combined

    def test_chunk_indices_sequential(self, chunker):
        content = "word " * 200
        chunks = chunker.chunk(content, content_type="text")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_python_chunking(self, chunker):
        code = (
            "import os\n\n"
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n"
        )
        chunks = chunker.chunk(code, content_type="python")
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "def foo" in all_text
        assert "def bar" in all_text

    def test_markdown_chunking(self, chunker):
        md = "# Section A\n\nSome text.\n\n## Section B\n\nMore text.\n"
        chunks = chunker.chunk(md, content_type="markdown")
        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Section A" in all_text

    def test_plain_sliding_window(self, chunker):
        words = ["word"] * 100
        content = " ".join(words)
        chunks = chunker.chunk(content, content_type="text")
        # With target=30 and overlap=5, we should get multiple chunks
        assert len(chunks) > 1

    def test_invalid_content_raises(self, chunker):
        from llmfs.core.exceptions import ChunkerError
        with pytest.raises(ChunkerError):
            chunker.chunk(12345)  # type: ignore

    def test_auto_detect_python(self, chunker):
        code = "def hello():\n    pass\n"
        chunks = chunker.chunk(code)
        assert len(chunks) >= 1

    def test_auto_detect_markdown(self, chunker):
        md = "# Title\n\nSome paragraph text here.\n"
        chunks = chunker.chunk(md)
        assert len(chunks) >= 1

    def test_offsets_within_bounds(self, chunker):
        content = "hello world foo bar baz " * 20
        chunks = chunker.chunk(content)
        for c in chunks:
            assert c.start_offset >= 0
            assert c.end_offset <= len(content)
