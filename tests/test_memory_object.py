"""Tests for llmfs.core.memory_object"""
import json
import pytest
from llmfs.core.memory_object import (
    Chunk, MemoryMetadata, MemoryObject, Relationship, SearchResult, Summaries,
)


class TestChunk:
    def test_roundtrip(self):
        c = Chunk(index=0, text="hello world", start_offset=0, end_offset=11, embedding_id="e1")
        assert Chunk.from_dict(c.to_dict()) == c

    def test_defaults(self):
        c = Chunk(index=1, text="x")
        assert c.embedding_id == ""
        assert c.summary == ""


class TestMemoryObject:
    def _make(self, **kwargs) -> MemoryObject:
        defaults = dict(id="abc", path="/test/path", content="hello", layer="knowledge")
        defaults.update(kwargs)
        return MemoryObject(**defaults)

    def test_content_hash_stable(self):
        obj = self._make(content="foo")
        assert obj.content_hash == obj.content_hash
        assert len(obj.content_hash) == 64

    def test_content_hash_changes(self):
        obj1 = self._make(content="foo")
        obj2 = self._make(content="bar")
        assert obj1.content_hash != obj2.content_hash

    def test_name_property(self):
        assert self._make(path="/a/b/c").name == "c"
        assert self._make(path="/single").name == "single"

    def test_tags_shortcut(self):
        obj = self._make()
        obj.metadata.tags = ["x", "y"]
        assert obj.tags == ["x", "y"]

    def test_to_dict_from_dict_roundtrip(self):
        obj = self._make(
            chunks=[Chunk(0, "chunk text", embedding_id="e0")],
            metadata=MemoryMetadata(tags=["t1"], source="agent"),
            relationships=[Relationship(target="/other", type="related_to", strength=0.9)],
        )
        restored = MemoryObject.from_dict(obj.to_dict())
        assert restored.id == obj.id
        assert restored.path == obj.path
        assert restored.layer == obj.layer
        assert len(restored.chunks) == 1
        assert restored.chunks[0].text == "chunk text"
        assert restored.relationships[0].target == "/other"
        assert "t1" in restored.tags

    def test_to_json_from_json_roundtrip(self):
        obj = self._make()
        restored = MemoryObject.from_json(obj.to_json())
        assert restored.id == obj.id

    def test_validate_bad_path(self):
        obj = self._make(path="no-slash")
        with pytest.raises(ValueError, match="must start with '/'"):
            obj.validate()

    def test_validate_bad_layer(self):
        obj = self._make(layer="invalid_layer")
        with pytest.raises(ValueError, match="Unknown layer"):
            obj.validate()

    def test_validate_empty_id(self):
        obj = self._make(id="")
        with pytest.raises(ValueError, match="id must not be empty"):
            obj.validate()

    def test_validate_ok(self):
        obj = self._make()
        obj.validate()  # must not raise

    def test_repr(self):
        assert "MemoryObject" in repr(self._make())


class TestSearchResult:
    def test_to_dict(self):
        r = SearchResult(path="/x", content="hello", score=0.9, tags=["a"])
        d = r.to_dict()
        assert d["path"] == "/x"
        assert d["score"] == 0.9
