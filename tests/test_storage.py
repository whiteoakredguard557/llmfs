"""Tests for llmfs.storage.metadata_db"""
import pytest
from llmfs.storage.metadata_db import MetadataDB


@pytest.fixture
def db(tmp_path):
    return MetadataDB(tmp_path / "test.db")


def _insert(db, path="/test/mem", layer="knowledge", id="abc123"):
    db.insert_file(id=id, path=path, name=path.rsplit("/", 1)[-1], layer=layer)


class TestMetadataDB:
    def test_insert_and_get(self, db):
        _insert(db)
        row = db.get_file("/test/mem")
        assert row is not None
        assert row["path"] == "/test/mem"
        assert row["layer"] == "knowledge"

    def test_get_nonexistent(self, db):
        assert db.get_file("/does/not/exist") is None

    def test_delete_file(self, db):
        _insert(db)
        n = db.delete_file("/test/mem")
        assert n == 1
        assert db.get_file("/test/mem") is None

    def test_update_file(self, db):
        _insert(db)
        db.update_file("/test/mem", size=512, content_hash="deadbeef")
        row = db.get_file("/test/mem")
        assert row["size"] == 512
        assert row["content_hash"] == "deadbeef"

    def test_touch_accessed(self, db):
        _insert(db)
        db.touch_accessed("/test/mem")
        row = db.get_file("/test/mem")
        assert row["accessed_at"] is not None

    def test_tags(self, db):
        _insert(db)
        row = db.get_file("/test/mem")
        db.tag_file(row["id"], "python")
        db.tag_file(row["id"], "debug")
        tags = db.get_tags_for_file(row["id"])
        assert "python" in tags
        assert "debug" in tags

    def test_untag(self, db):
        _insert(db)
        row = db.get_file("/test/mem")
        db.tag_file(row["id"], "python")
        db.untag_file(row["id"], "python")
        assert "python" not in db.get_tags_for_file(row["id"])

    def test_set_tags_replaces(self, db):
        _insert(db)
        row = db.get_file("/test/mem")
        db.tag_file(row["id"], "old")
        db.set_tags(row["id"], ["new1", "new2"])
        tags = db.get_tags_for_file(row["id"])
        assert "old" not in tags
        assert set(tags) == {"new1", "new2"}

    def test_list_files_by_layer(self, db):
        _insert(db, path="/a", layer="knowledge", id="id1")
        _insert(db, path="/b", layer="events", id="id2")
        knowledge = db.list_files(layer="knowledge")
        assert len(knowledge) == 1
        assert knowledge[0]["path"] == "/a"

    def test_list_files_by_prefix(self, db):
        _insert(db, path="/projects/x", id="id1")
        _insert(db, path="/knowledge/y", id="id2")
        results = db.list_files(path_prefix="/projects")
        assert len(results) == 1
        assert results[0]["path"] == "/projects/x"

    def test_chunks_crud(self, db):
        _insert(db)
        row = db.get_file("/test/mem")
        db.insert_chunk(
            id="c1", file_id=row["id"], chunk_index=0,
            start_offset=0, end_offset=10, text="hello", embedding_id="emb1",
        )
        chunks = db.get_chunks(row["id"])
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello"

    def test_delete_chunks(self, db):
        _insert(db)
        row = db.get_file("/test/mem")
        db.insert_chunk(id="c1", file_id=row["id"], chunk_index=0,
                        start_offset=0, end_offset=5, text="hi", embedding_id="e1")
        db.delete_chunks(row["id"])
        assert db.get_chunks(row["id"]) == []

    def test_relationships(self, db):
        _insert(db, path="/src", id="id1")
        _insert(db, path="/tgt", id="id2")
        db.insert_relationship(id="r1", source_id="id1", target_id="id2",
                               rel_type="related_to", strength=0.9)
        rels = db.get_relationships("id1")
        assert len(rels) == 1
        assert rels[0]["type"] == "related_to"
        assert abs(rels[0]["strength"] - 0.9) < 0.001

    def test_expire_ttl(self, db):
        from datetime import datetime, timedelta, timezone
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        db.insert_file(id="exp1", path="/expired", name="expired",
                       layer="short_term", ttl_expires=past)
        deleted = db.expire_ttl()
        assert deleted >= 1
        assert db.get_file("/expired") is None

    def test_search_cache(self, db):
        db.cache_set("hash1", [{"path": "/x", "score": 0.9}], ttl_seconds=300)
        cached = db.cache_get("hash1")
        assert cached is not None
        assert cached[0]["path"] == "/x"

    def test_cache_miss_returns_none(self, db):
        assert db.cache_get("nonexistent") is None
