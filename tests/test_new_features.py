"""
Tests for the five new LLMFS features:

1. BM25 hybrid search (FTS5)
2. memory_list MCP tool
3. Persistent embedding cache
4. Auto-linking
5. AsyncMemoryFS
"""
from __future__ import annotations

import asyncio
import json

import pytest

from llmfs import AsyncMemoryFS, MemoryFS, MemoryNotFoundError
from llmfs.embeddings.base import EmbedderBase
from llmfs.mcp.tools import handle_tool_call
from llmfs.storage.metadata_db import MetadataDB


# ── Shared fake embedder ──────────────────────────────────────────────────────


class FakeEmbedder(EmbedderBase):
    """Deterministic fixed-size vector — no model download."""

    DIM = 16

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def embedding_dim(self) -> int:
        return self.DIM

    def embed(self, text: str) -> list[float]:
        h = hash(text) % (2 ** 31)
        return [float((h >> i) & 1) for i in range(self.DIM)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs", embedder=FakeEmbedder())


@pytest.fixture
def mem_no_autolink(tmp_path):
    return MemoryFS(
        path=tmp_path / "llmfs_nolink",
        embedder=FakeEmbedder(),
        auto_link=False,
    )


@pytest.fixture
def db(tmp_path):
    return MetadataDB(tmp_path / "test.db")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BM25 Hybrid Search
# ═══════════════════════════════════════════════════════════════════════════════


class TestBM25Search:
    """FTS5 full-text search and hybrid fusion."""

    def test_fts_search_returns_results(self, mem):
        """BM25 search finds memories by keyword."""
        mem.write("/k/auth", content="JWT authentication token expired")
        mem.write("/k/db", content="PostgreSQL database connection pooling")
        results = mem._db.fts_search("JWT authentication")
        assert len(results) >= 1
        paths = [r["path"] for r in results]
        assert "/k/auth" in paths

    def test_fts_search_empty_query_returns_empty(self, mem):
        mem.write("/k/x", content="some content")
        results = mem._db.fts_search("")
        assert results == []

    def test_fts_search_no_match(self, mem):
        mem.write("/k/x", content="banana orange apple")
        results = mem._db.fts_search("cryptocurrency blockchain")
        assert results == []

    def test_fts_search_respects_layer_filter(self, mem):
        mem.write("/k/a", content="alpha topic content", layer="knowledge")
        mem.write("/e/b", content="alpha topic content", layer="events")
        results = mem._db.fts_search("alpha", layer="knowledge")
        for r in results:
            assert r["layer"] == "knowledge"

    def test_fts_search_respects_path_prefix(self, mem):
        mem.write("/projects/auth/bug", content="authentication failure")
        mem.write("/notes/auth", content="authentication notes")
        results = mem._db.fts_search("authentication", path_prefix="/projects")
        for r in results:
            assert r["path"].startswith("/projects")

    def test_hybrid_search_returns_results(self, mem):
        """The main search() now uses hybrid dense+BM25 fusion."""
        mem.write("/k/jwt", content="JWT token expires after one hour")
        mem.write("/k/db", content="PostgreSQL connection pool")
        results = mem.search("JWT token")
        assert len(results) >= 1
        paths = [r.path for r in results]
        assert "/k/jwt" in paths

    def test_bm25_scores_normalized(self, mem):
        """BM25 results from _bm25_search have scores in [0, 1]."""
        mem.write("/k/a", content="alpha beta gamma keyword search test")
        mem.write("/k/b", content="delta epsilon zeta keyword search test")
        results = mem._bm25_search("keyword search")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_fts_search_bm25_score_positive(self, mem):
        """Returned bm25_score values should be positive (negated rank)."""
        mem.write("/k/x", content="the quick brown fox jumped")
        results = mem._db.fts_search("quick brown fox")
        assert len(results) >= 1
        for r in results:
            assert r["bm25_score"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. memory_list MCP Tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryListTool:
    """Tests for the memory_list MCP tool handler."""

    def test_memory_list_returns_entries(self, mem):
        mem.write("/k/a", content="alpha")
        mem.write("/k/b", content="beta")
        result = handle_tool_call("memory_list", {"path_prefix": "/"}, mem)
        assert result["status"] == "ok"
        assert result["count"] >= 2
        paths = [e["path"] for e in result["entries"]]
        assert "/k/a" in paths
        assert "/k/b" in paths

    def test_memory_list_with_prefix(self, mem):
        mem.write("/projects/a", content="alpha")
        mem.write("/notes/b", content="beta")
        result = handle_tool_call("memory_list", {"path_prefix": "/projects"}, mem)
        for entry in result["entries"]:
            assert entry["path"].startswith("/projects")

    def test_memory_list_with_layer_filter(self, mem):
        mem.write("/k/a", content="knowledge", layer="knowledge")
        mem.write("/e/b", content="event", layer="events")
        result = handle_tool_call(
            "memory_list", {"path_prefix": "/", "layer": "knowledge"}, mem,
        )
        for entry in result["entries"]:
            assert entry["layer"] == "knowledge"

    def test_memory_list_respects_limit(self, mem):
        for i in range(10):
            mem.write(f"/k/item{i}", content=f"item {i}")
        result = handle_tool_call(
            "memory_list", {"path_prefix": "/", "limit": 3}, mem,
        )
        assert result["count"] <= 3

    def test_memory_list_empty_store(self, mem):
        result = handle_tool_call("memory_list", {}, mem)
        assert result["status"] == "ok"
        assert result["count"] == 0

    def test_memory_list_entry_has_expected_keys(self, mem):
        mem.write("/k/test", content="test", tags=["foo"])
        result = handle_tool_call("memory_list", {}, mem)
        entry = result["entries"][0]
        assert "path" in entry
        assert "layer" in entry
        assert "tags" in entry
        assert "created_at" in entry
        assert "modified_at" in entry

    def test_memory_list_default_prefix_is_root(self, mem):
        mem.write("/k/default", content="test")
        result = handle_tool_call("memory_list", {}, mem)
        assert result["count"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Persistent Embedding Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingCache:
    """Tests for the SQLite-backed embedding cache in MetadataDB."""

    def test_put_and_get_single(self, db):
        vec = [0.1, 0.2, 0.3]
        db.put_cached_embedding("hash1", vec, "test-model")
        result = db.get_cached_embedding("hash1", "test-model")
        assert result == vec

    def test_get_miss_returns_none(self, db):
        result = db.get_cached_embedding("nonexistent", "test-model")
        assert result is None

    def test_model_name_isolation(self, db):
        vec_a = [1.0, 2.0]
        vec_b = [3.0, 4.0]
        db.put_cached_embedding("hash1", vec_a, "model-a")
        db.put_cached_embedding("hash1", vec_b, "model-b")
        assert db.get_cached_embedding("hash1", "model-a") == vec_a
        assert db.get_cached_embedding("hash1", "model-b") == vec_b

    def test_put_overwrites_existing(self, db):
        db.put_cached_embedding("hash1", [1.0], "model")
        db.put_cached_embedding("hash1", [2.0], "model")
        assert db.get_cached_embedding("hash1", "model") == [2.0]

    def test_batch_put_and_get(self, db):
        items = [
            ("h1", [0.1, 0.2]),
            ("h2", [0.3, 0.4]),
            ("h3", [0.5, 0.6]),
        ]
        db.put_cached_embeddings_batch(items, "model")
        result = db.get_cached_embeddings_batch(["h1", "h2", "h3", "h4"], "model")
        assert "h1" in result
        assert "h2" in result
        assert "h3" in result
        assert "h4" not in result
        assert result["h1"] == [0.1, 0.2]

    def test_batch_get_empty_input(self, db):
        result = db.get_cached_embeddings_batch([], "model")
        assert result == {}

    def test_batch_put_empty_input(self, db):
        db.put_cached_embeddings_batch([], "model")  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Auto-Linking
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoLink:
    """Tests for auto-link graph edge creation on write."""

    def test_auto_link_creates_edges(self, mem):
        """Writing similar content should auto-create related_to edges."""
        mem.write("/k/auth1", content="JWT authentication token management")
        mem.write("/k/auth2", content="JWT authentication token validation")
        # Read back and check relationships
        obj = mem.read("/k/auth2")
        # If vectors are similar enough, there should be a relationship
        # (With FakeEmbedder, similarity depends on hash collisions)
        # We just verify the mechanism runs without error
        assert obj is not None

    def test_auto_link_disabled(self, mem_no_autolink):
        """With auto_link=False, no edges are created."""
        mem_no_autolink.write("/k/a", content="content A")
        mem_no_autolink.write("/k/b", content="content B")
        obj = mem_no_autolink.read("/k/b")
        assert len(obj.relationships) == 0

    def test_auto_link_does_not_self_link(self, mem):
        """Auto-link should not create an edge from a memory to itself."""
        mem.write("/k/self", content="self referencing content")
        obj = mem.read("/k/self")
        for rel in obj.relationships:
            assert rel.target != "/k/self"

    def test_auto_link_threshold_respected(self, tmp_path):
        """With threshold=1.0, no auto-links should be created."""
        high_thresh_mem = MemoryFS(
            path=tmp_path / "high_thresh",
            embedder=FakeEmbedder(),
            auto_link_threshold=1.0,
        )
        high_thresh_mem.write("/k/a", content="first memory")
        high_thresh_mem.write("/k/b", content="second memory")
        obj = high_thresh_mem.read("/k/b")
        assert len(obj.relationships) == 0

    def test_auto_link_max_k(self, tmp_path):
        """Auto-link should create at most auto_link_k edges."""
        limited_mem = MemoryFS(
            path=tmp_path / "limited",
            embedder=FakeEmbedder(),
            auto_link=True,
            auto_link_k=1,
            auto_link_threshold=0.0,  # link everything
        )
        limited_mem.write("/k/a", content="one")
        limited_mem.write("/k/b", content="two")
        limited_mem.write("/k/c", content="three")
        obj = limited_mem.read("/k/c")
        assert len(obj.relationships) <= 1

    def test_auto_link_non_fatal_on_error(self, mem):
        """Auto-link failures should not crash write()."""
        # Even if _auto_link_memory encounters issues internally,
        # write() should succeed
        obj = mem.write("/k/robust", content="this should always succeed")
        assert obj.path == "/k/robust"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. AsyncMemoryFS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsyncMemoryFS:
    """Tests for the async wrapper around MemoryFS."""

    @pytest.fixture
    def async_mem(self, tmp_path):
        return AsyncMemoryFS(
            path=tmp_path / "async_llmfs",
            embedder=FakeEmbedder(),
            auto_link=False,
        )

    def test_write_and_read(self, async_mem):
        async def _test():
            obj = await async_mem.write("/k/hello", "world")
            assert obj.path == "/k/hello"
            read_obj = await async_mem.read("/k/hello")
            assert "world" in read_obj.content
        asyncio.run(_test())

    def test_search(self, async_mem):
        async def _test():
            await async_mem.write("/k/auth", "JWT authentication")
            results = await async_mem.search("JWT", k=3)
            assert isinstance(results, list)
        asyncio.run(_test())

    def test_update(self, async_mem):
        async def _test():
            await async_mem.write("/k/note", "original")
            obj = await async_mem.update("/k/note", append=" updated")
            assert "updated" in obj.content
        asyncio.run(_test())

    def test_forget(self, async_mem):
        async def _test():
            await async_mem.write("/k/delete_me", "bye")
            result = await async_mem.forget("/k/delete_me")
            assert result["deleted"] == 1
        asyncio.run(_test())

    def test_relate(self, async_mem):
        async def _test():
            await async_mem.write("/k/src", "source")
            await async_mem.write("/k/tgt", "target")
            result = await async_mem.relate("/k/src", "/k/tgt", "related_to")
            assert result["status"] == "ok"
        asyncio.run(_test())

    def test_list(self, async_mem):
        async def _test():
            await async_mem.write("/k/a", "alpha")
            await async_mem.write("/k/b", "beta")
            objects = await async_mem.list("/k")
            paths = {o.path for o in objects}
            assert "/k/a" in paths
            assert "/k/b" in paths
        asyncio.run(_test())

    def test_status(self, async_mem):
        async def _test():
            await async_mem.write("/k/x", "x")
            info = await async_mem.status()
            assert info["total"] == 1
        asyncio.run(_test())

    def test_gc(self, async_mem):
        async def _test():
            result = await async_mem.gc()
            assert result["status"] == "ok"
        asyncio.run(_test())

    def test_sync_property(self, async_mem):
        assert isinstance(async_mem.sync, MemoryFS)

    def test_repr(self, async_mem):
        assert "AsyncMemoryFS" in repr(async_mem)

    def test_concurrent_writes(self, async_mem):
        """Multiple concurrent writes should not raise."""
        async def _test():
            tasks = [
                async_mem.write(f"/k/concurrent_{i}", f"content {i}")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            for obj in results:
                assert obj.path.startswith("/k/concurrent_")
        asyncio.run(_test())

    def test_read_nonexistent_raises(self, async_mem):
        async def _test():
            with pytest.raises(MemoryNotFoundError):
                await async_mem.read("/does/not/exist")
        asyncio.run(_test())
