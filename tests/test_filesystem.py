"""
Tests for llmfs.core.filesystem (MemoryFS).

The LocalEmbedder loads a 22MB model, so we mock the embedder with a fast
deterministic stub that returns fixed-dimension vectors.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from llmfs import MemoryFS, MemoryNotFoundError
from llmfs.embeddings.base import EmbedderBase


class FakeEmbedder(EmbedderBase):
    """Returns a deterministic fixed-size vector — no model download."""

    DIM = 16

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def embedding_dim(self) -> int:
        return self.DIM

    def embed(self, text: str) -> list[float]:
        # Deterministic: hash the text to produce a stable vector
        h = hash(text) % (2 ** 31)
        return [float((h >> i) & 1) for i in range(self.DIM)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs", embedder=FakeEmbedder())


class TestMemoryFSWrite:
    def test_write_returns_memory_object(self, mem):
        obj = mem.write("/test/hello", content="Hello, world!")
        assert obj.path == "/test/hello"
        assert obj.layer == "knowledge"
        assert len(obj.chunks) >= 1

    def test_write_creates_chunks(self, mem):
        obj = mem.write("/test/code", content="def foo():\n    return 1\n")
        assert len(obj.chunks) >= 1

    def test_write_populates_summaries(self, mem):
        """write() should populate summaries.level_1 and summaries.level_2."""
        content = (
            "The authentication module uses JWT tokens for session management. "
            "Tokens expire after one hour to reduce the attack surface. "
            "The refresh endpoint accepts a valid refresh token and issues a new access token."
        )
        obj = mem.write("/test/summaries", content=content)
        assert isinstance(obj.summaries.level_2, str)
        assert len(obj.summaries.level_2) > 0
        assert isinstance(obj.summaries.level_1, list)
        assert len(obj.summaries.level_1) == len(obj.chunks)

    def test_write_with_tags(self, mem):
        obj = mem.write("/test/tagged", content="tagged memory", tags=["a", "b"])
        assert "a" in obj.tags
        assert "b" in obj.tags

    def test_write_with_layer(self, mem):
        obj = mem.write("/test/event", content="something happened", layer="events")
        assert obj.layer == "events"

    def test_write_invalid_path_raises(self, mem):
        with pytest.raises(ValueError, match="must start with '/'"):
            mem.write("no-slash", content="x")

    def test_write_invalid_layer_raises(self, mem):
        with pytest.raises(ValueError, match="Invalid layer"):
            mem.write("/test/x", content="x", layer="nonsense")

    def test_write_same_content_no_reembed(self, mem):
        mem.write("/test/dup", content="same content")
        obj2 = mem.write("/test/dup", content="same content")
        # Should return the object without error
        assert obj2.path == "/test/dup"

    def test_write_update_content(self, mem):
        mem.write("/test/update", content="v1")
        obj = mem.write("/test/update", content="v2 changed")
        assert obj.path == "/test/update"


class TestMemoryFSRead:
    def test_read_returns_object(self, mem):
        mem.write("/k/note", content="important note")
        obj = mem.read("/k/note")
        assert obj.path == "/k/note"
        assert "important note" in obj.content

    def test_read_nonexistent_raises(self, mem):
        with pytest.raises(MemoryNotFoundError):
            mem.read("/does/not/exist")

    def test_read_with_query(self, mem):
        mem.write("/k/long", content="paragraph one about auth.\n\nparagraph two about db.")
        obj = mem.read("/k/long", query="auth")
        assert obj is not None  # just verify it doesn't crash


class TestMemoryFSSearch:
    def test_search_returns_results(self, mem):
        mem.write("/k/auth", content="JWT authentication logic", tags=["auth"])
        mem.write("/k/db", content="PostgreSQL database connection")
        results = mem.search("authentication")
        assert len(results) >= 1
        paths = [r.path for r in results]
        assert "/k/auth" in paths

    def test_search_respects_k(self, mem):
        for i in range(10):
            mem.write(f"/k/item{i}", content=f"item {i} content about topic")
        results = mem.search("topic", k=3)
        assert len(results) <= 3

    def test_search_layer_filter(self, mem):
        mem.write("/k/knowledge_item", content="knowledge content", layer="knowledge")
        mem.write("/e/event_item", content="event content", layer="events")
        results = mem.search("content", layer="knowledge")
        for r in results:
            assert r.metadata.get("layer") == "knowledge"

    def test_search_tag_filter(self, mem):
        mem.write("/k/a", content="python code example", tags=["python"])
        mem.write("/k/b", content="java code example", tags=["java"])
        results = mem.search("code", tags=["python"])
        for r in results:
            assert "python" in r.tags

    def test_search_empty_store_returns_empty(self, mem):
        results = mem.search("anything")
        assert results == []


class TestMemoryFSUpdate:
    def test_update_append(self, mem):
        mem.write("/k/note", content="original content")
        obj = mem.update("/k/note", append="appended text")
        assert "appended" in obj.content

    def test_update_replace(self, mem):
        mem.write("/k/note", content="old content")
        obj = mem.update("/k/note", content="brand new content")
        assert "brand new" in obj.content

    def test_update_tags(self, mem):
        mem.write("/k/note", content="note", tags=["old"])
        obj = mem.update("/k/note", tags_add=["new"], tags_remove=["old"])
        assert "new" in obj.tags
        assert "old" not in obj.tags

    def test_update_nonexistent_raises(self, mem):
        with pytest.raises(MemoryNotFoundError):
            mem.update("/missing", content="x")

    def test_update_both_content_and_append_raises(self, mem):
        mem.write("/k/note", content="x")
        with pytest.raises(ValueError):
            mem.update("/k/note", content="new", append="extra")


class TestMemoryFSForget:
    def test_forget_by_path(self, mem):
        mem.write("/k/delete_me", content="bye")
        result = mem.forget("/k/delete_me")
        assert result["deleted"] == 1
        with pytest.raises(MemoryNotFoundError):
            mem.read("/k/delete_me")

    def test_forget_nonexistent_raises(self, mem):
        with pytest.raises(MemoryNotFoundError):
            mem.forget("/nope")

    def test_forget_by_layer(self, mem):
        mem.write("/s/1", content="session 1", layer="session")
        mem.write("/s/2", content="session 2", layer="session")
        mem.write("/k/1", content="knowledge 1", layer="knowledge")
        result = mem.forget(layer="session")
        assert result["deleted"] == 2

    def test_forget_no_args_raises(self, mem):
        with pytest.raises(ValueError):
            mem.forget()


class TestMemoryFSRelate:
    def test_relate_creates_relationship(self, mem):
        mem.write("/src", content="source")
        mem.write("/tgt", content="target")
        result = mem.relate("/src", "/tgt", "related_to")
        assert result["status"] == "ok"
        assert "relationship_id" in result

    def test_relate_missing_source_raises(self, mem):
        mem.write("/tgt", content="target")
        with pytest.raises(MemoryNotFoundError):
            mem.relate("/missing", "/tgt", "related_to")

    def test_relate_target_is_path_not_uuid(self, mem):
        """Relationship.target must be a memory path, not an internal UUID."""
        mem.write("/src", content="source memory")
        mem.write("/tgt", content="target memory")
        mem.relate("/src", "/tgt", "related_to")
        obj = mem.read("/src")
        assert len(obj.relationships) == 1
        assert obj.relationships[0].target == "/tgt"
        # Must not be a UUID (UUIDs are 36-char hex with dashes)
        assert not obj.relationships[0].target.replace("-", "").isalnum() or obj.relationships[0].target.startswith("/")



class TestMemoryFSList:
    def test_list_all(self, mem):
        mem.write("/a/1", content="one")
        mem.write("/a/2", content="two")
        mem.write("/b/3", content="three")
        objects = mem.list("/")
        paths = {o.path for o in objects}
        assert "/a/1" in paths
        assert "/b/3" in paths

    def test_list_with_prefix(self, mem):
        mem.write("/a/1", content="one")
        mem.write("/b/2", content="two")
        objects = mem.list("/a")
        assert all(o.path.startswith("/a") for o in objects)


class TestMemoryFSStatus:
    def test_status_keys(self, mem):
        mem.write("/k/x", content="x")
        info = mem.status()
        assert "total" in info
        assert "layers" in info
        assert "chunks" in info
        assert "disk_mb" in info

    def test_status_total_increments(self, mem):
        assert mem.status()["total"] == 0
        mem.write("/k/a", content="a")
        assert mem.status()["total"] == 1


class TestMemoryFSGC:
    def test_gc_returns_dict(self, mem):
        result = mem.gc()
        assert "deleted" in result
        assert result["status"] == "ok"

    def test_gc_removes_expired_memories(self, mem):
        from datetime import datetime, timedelta, timezone
        # Write a memory with a 1-minute TTL
        mem.write("/tmp/expire", content="will expire", layer="short_term", ttl_minutes=1)
        assert mem.status()["total"] == 1
        # Manually backdate the ttl_expires to simulate expiry
        row = mem._db.get_file("/tmp/expire")
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mem._db._exec(
            "UPDATE files SET ttl_expires = ? WHERE id = ?",
            (past, row["id"]),
        )
        result = mem.gc()
        assert result["deleted"] == 1
        assert mem.status()["total"] == 0

