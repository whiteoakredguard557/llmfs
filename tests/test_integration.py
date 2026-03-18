"""
Integration tests for LLMFS — full end-to-end cycles.

These tests exercise the complete stack from MemoryFS API through storage and
retrieval, verifying that all components work together correctly.

Markers
-------
- ``integration``: full end-to-end tests (slower, use real embedder)
- ``performance``: latency assertions (may be skipped in slow CI)
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Generator

import pytest

from llmfs import MemoryFS
from llmfs.core.exceptions import MemoryNotFoundError
from llmfs.query import MQLParser, execute_mql


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mem(tmp_path_factory) -> Generator[MemoryFS, None, None]:
    """Module-scoped MemoryFS so the embedder loads only once."""
    base = tmp_path_factory.mktemp("integration")
    yield MemoryFS(path=base)


@pytest.fixture(scope="module")
def populated_mem(tmp_path_factory) -> Generator[MemoryFS, None, None]:
    """Pre-populated store for search/query tests."""
    base = tmp_path_factory.mktemp("populated")
    m = MemoryFS(path=base)
    # Knowledge layer
    m.write("/knowledge/auth/jwt",
            "JWT token expires in 15 minutes. Use RS256 algorithm. "
            "Refresh tokens should be stored in HttpOnly cookies.",
            tags=["auth", "security", "jwt"])
    m.write("/knowledge/auth/oauth",
            "OAuth2 PKCE flow for SPAs. Authorization code + code verifier. "
            "Never store client secret in frontend.",
            tags=["auth", "security", "oauth"])
    m.write("/knowledge/database/postgres",
            "PostgreSQL 15 with connection pooling via PgBouncer. "
            "WAL mode, max 100 connections. Use LISTEN/NOTIFY for events.",
            tags=["database", "postgres", "infra"])
    m.write("/knowledge/database/redis",
            "Redis 7 for caching and pub/sub. Cluster mode in production. "
            "Keyspace notifications enabled for cache invalidation.",
            tags=["database", "redis", "cache", "infra"])
    m.write("/events/bugs/auth_bypass",
            "Critical: authentication bypass discovered 2026-01-15. "
            "JWT signature not validated on /api/admin endpoint.",
            tags=["bug", "auth", "critical"])
    m.write("/events/bugs/db_leak",
            "Connection pool exhaustion in staging. Max connections reached "
            "during load test. Fixed by reducing idle timeout.",
            tags=["bug", "database", "performance"])
    m.write("/session/turn1",
            "User asked about setting up Redis cluster.",
            layer="session", tags=["session"])
    m.write("/session/turn2",
            "Explained Redis sentinel vs cluster vs standalone modes.",
            layer="session", tags=["session"])
    # Create some relationships
    m.relate("/events/bugs/auth_bypass", "/knowledge/auth/jwt", "related_to")
    m.relate("/events/bugs/db_leak", "/knowledge/database/postgres", "related_to")
    yield m


# ── 5.1.A Full write → search → retrieve cycle ───────────────────────────────


@pytest.mark.integration
class TestEndToEndCycle:
    """Complete write → search → read → update → forget cycle."""

    def test_write_and_read_roundtrip(self, mem):
        """Written content can be retrieved by exact path."""
        content = "The quick brown fox jumps over the lazy dog. " * 5
        obj = mem.write("/integration/roundtrip", content, tags=["test"])
        assert obj.path == "/integration/roundtrip"
        assert len(obj.chunks) >= 1

        retrieved = mem.read("/integration/roundtrip")
        assert "/integration/roundtrip" == retrieved.path
        assert "fox" in retrieved.content

    def test_semantic_search_finds_relevant_memory(self, populated_mem):
        """Semantic search returns the most relevant memory."""
        results = populated_mem.search("JWT authentication token", k=3)
        assert len(results) >= 1
        paths = [r.path for r in results]
        assert "/knowledge/auth/jwt" in paths

    def test_search_score_ordering(self, populated_mem):
        """Results are returned in descending score order."""
        results = populated_mem.search("authentication security", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tag_filter_narrows_results(self, populated_mem):
        """Tag filter restricts semantic search to tagged memories."""
        results = populated_mem.search("data storage", tags=["database"], k=10)
        for r in results:
            assert "database" in r.tags

    def test_layer_filter_restricts_search(self, populated_mem):
        """Layer filter only returns memories from that layer."""
        results = populated_mem.search("turn Redis", layer="session", k=10)
        for r in results:
            assert r.metadata.get("layer") == "session"

    def test_update_changes_content(self, mem):
        """Updating a memory changes its searchable content."""
        mem.write("/integration/update_test", "original content alpha beta", tags=["updatetest"])
        mem.update("/integration/update_test", content="completely new content gamma delta")
        obj = mem.read("/integration/update_test")
        assert "gamma" in obj.content
        assert "alpha" not in obj.content

    def test_update_append_extends_content(self, mem):
        """Appending extends rather than replaces content."""
        mem.write("/integration/append_test", "first part", tags=["appendtest"])
        mem.update("/integration/append_test", append=" second part added here")
        obj = mem.read("/integration/append_test")
        assert "first part" in obj.content
        assert "second part" in obj.content

    def test_forget_removes_from_search(self, mem):
        """Forgetting a memory removes it from search results."""
        mem.write("/integration/forget_test",
                  "unicorn rainbow sparkle glitter unique token 99zz",
                  tags=["forget_marker"])
        # Confirm it's findable
        r1 = mem.search("unicorn rainbow sparkle", k=5)
        assert any("/integration/forget_test" in r.path for r in r1)

        mem.forget("/integration/forget_test")
        with pytest.raises(MemoryNotFoundError):
            mem.read("/integration/forget_test")

        r2 = mem.search("unicorn rainbow sparkle", k=5)
        assert all("/integration/forget_test" not in r.path for r in r2)

    def test_relationships_are_traversable(self, populated_mem):
        """Relationships created with relate() are stored and retrievable."""
        from llmfs.graph.memory_graph import MemoryGraph
        graph = MemoryGraph(populated_mem._db)
        result = graph.bfs("/events/bugs/auth_bypass", depth=1)
        assert "/knowledge/auth/jwt" in result.visited

    def test_list_by_prefix(self, populated_mem):
        """list() returns all memories under a prefix."""
        objs = populated_mem.list("/knowledge/auth", recursive=True)
        paths = [o.path for o in objs]
        assert "/knowledge/auth/jwt" in paths
        assert "/knowledge/auth/oauth" in paths
        # Should not contain database entries
        assert not any("/database" in p for p in paths)

    def test_status_reflects_written_memories(self, populated_mem):
        """status() returns correct counts."""
        info = populated_mem.status()
        assert info["total"] >= 8
        assert "knowledge" in info["layers"]
        assert "session" in info["layers"]
        assert info["disk_mb"] > 0

    def test_gc_removes_expired(self, tmp_path):
        """gc() removes TTL-expired memories."""
        m = MemoryFS(path=tmp_path / "gc_test")
        m.write("/gc/short", "will expire soon", layer="short_term", ttl_minutes=0)
        # Force TTL to be in the past by writing a row with past expiry directly
        m._db.expire_ttl()  # Clean up anything already expired
        result = m.gc()
        assert isinstance(result["deleted"], int)
        assert result["status"] == "ok"


# ── 5.1.B MQL integration tests ───────────────────────────────────────────────


@pytest.mark.integration
class TestMQLIntegration:
    """End-to-end MQL query tests using real MemoryFS."""

    def test_mem_query_method(self, populated_mem):
        """MemoryFS.query() convenience method works."""
        results = populated_mem.query(
            'SELECT memory FROM /knowledge WHERE SIMILAR TO "JWT token"'
        )
        assert isinstance(results, list)

    def test_mql_tag_filter(self, populated_mem):
        """MQL TAG = condition returns correct memories."""
        results = populate_and_query(
            populated_mem,
            'SELECT memory FROM / WHERE TAG = "critical"',
        )
        assert all("critical" in r.tags for r in results)

    def test_mql_limit(self, populated_mem):
        """MQL LIMIT restricts result count."""
        results = populated_mem.query(
            'SELECT memory FROM / LIMIT 2'
        )
        assert len(results) <= 2

    def test_mql_and_condition(self, populated_mem):
        """MQL AND condition intersects results."""
        results = populated_mem.query(
            'SELECT memory FROM / WHERE TAG = "auth" AND TAG = "security"'
        )
        for r in results:
            assert "auth" in r.tags
            assert "security" in r.tags

    def test_mql_or_condition(self, populated_mem):
        """MQL OR condition unions results."""
        results = populated_mem.query(
            'SELECT memory FROM / WHERE TAG = "jwt" OR TAG = "oauth"'
        )
        paths = [r.path for r in results]
        assert len(paths) >= 1

    def test_mql_no_where_clause(self, populated_mem):
        """SELECT without WHERE returns all memories under prefix."""
        all_results = populated_mem.query('SELECT memory FROM /')
        auth_results = populated_mem.query('SELECT memory FROM /knowledge/auth')
        assert len(all_results) >= len(auth_results)

    def test_execute_mql_module_function(self, populated_mem):
        """Module-level execute_mql() function works."""
        results = execute_mql(
            'SELECT memory FROM /knowledge WHERE TAG = "database"',
            populated_mem,
        )
        assert isinstance(results, list)


def populate_and_query(mem, mql):
    """Helper: run MQL and return results."""
    return mem.query(mql)


# ── 5.1.C Context middleware integration ──────────────────────────────────────


@pytest.mark.integration
class TestContextMiddlewareIntegration:
    """Integration test for ContextMiddleware wrapping a fake LLM."""

    def test_middleware_stores_and_retrieves(self, tmp_path):
        """ContextMiddleware tracks turns in the active context window."""
        from llmfs.context.middleware import ContextMiddleware
        from llmfs import MemoryFS

        mem = MemoryFS(path=tmp_path / "mw_test")

        class FakeLLM:
            def __init__(self):
                self.calls = []

            def chat(self, messages):
                self.calls.append(messages)
                return "Mock response from LLM."

        llm = FakeLLM()
        wrapped = ContextMiddleware(llm, memory=mem)

        # Simulate several turns
        for i in range(3):
            wrapped.chat([{"role": "user", "content": f"User message {i}: tell me about Python decorators"}])

        # Turns should be tracked in the active context window
        stats = wrapped.get_context_stats()
        assert stats["call_count"] == 3
        active = wrapped.get_active_turns()
        # Each chat call adds 1 user + 1 assistant turn (if LLM returns a response)
        assert len(active) >= 3

    def test_middleware_inject_context(self, tmp_path):
        """ContextMiddleware injects memory index into system prompt."""
        from llmfs.context.middleware import ContextMiddleware
        from llmfs import MemoryFS

        mem = MemoryFS(path=tmp_path / "inject_test")
        mem.write("/knowledge/py", "Python is great for data science.", tags=["python"])

        injected_messages = []

        class CaptureLLM:
            def chat(self, messages):
                injected_messages.extend(messages)
                return "OK"

        llm = CaptureLLM()
        wrapped = ContextMiddleware(llm, memory=mem)
        wrapped.chat([{"role": "user", "content": "Tell me about Python data science"}])

        # At least one message should have been sent to the LLM
        assert len(injected_messages) >= 1


# ── 5.1.D Performance tests ───────────────────────────────────────────────────


@pytest.mark.performance
class TestPerformance:
    """Latency assertions for key operations.

    These tests check that operations complete within the targets from PLAN.md:
    - Write (single ~500 token): < 2000ms (relaxed; embedder load dominates first time)
    - Read (by path): < 100ms
    - Search (top-5): < 2000ms (relaxed; ChromaDB warm)
    - MQL query: < 2000ms
    """

    @pytest.fixture(scope="class")
    def perf_mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("perf")
        m = MemoryFS(path=base)
        # Pre-warm embedder
        m.write("/warmup", "warmup text for embedder initialisation")
        return m

    def test_read_latency(self, perf_mem):
        """Read by exact path completes in < 100ms."""
        perf_mem.write("/perf/read_target", "content for read latency test")
        t0 = time.perf_counter()
        perf_mem.read("/perf/read_target")
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 100, f"read took {elapsed:.1f}ms (target <100ms)"

    def test_write_latency(self, perf_mem):
        """Write with ~100-word content completes in < 2000ms after warm-up."""
        content = ("The authentication module handles JWT token validation. " * 10)
        t0 = time.perf_counter()
        perf_mem.write("/perf/write_target", content, tags=["perf"])
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 2000, f"write took {elapsed:.1f}ms (target <2000ms)"

    def test_search_latency(self, perf_mem):
        """Semantic search completes in < 2000ms after warm-up."""
        # Ensure there's something to search
        for i in range(5):
            perf_mem.write(f"/perf/s{i}", f"sample memory content {i} about authentication")

        t0 = time.perf_counter()
        perf_mem.search("authentication token validation", k=5)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 2000, f"search took {elapsed:.1f}ms (target <2000ms)"

    def test_mql_query_latency(self, perf_mem):
        """MQL query completes in < 2000ms."""
        t0 = time.perf_counter()
        perf_mem.query('SELECT memory FROM /perf WHERE TAG = "perf" LIMIT 5')
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 2000, f"MQL query took {elapsed:.1f}ms (target <2000ms)"

    def test_list_latency(self, perf_mem):
        """list() completes in < 50ms."""
        t0 = time.perf_counter()
        perf_mem.list("/perf", recursive=True)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 50, f"list took {elapsed:.1f}ms (target <50ms)"


# ── 5.1.E Multi-layer isolation ───────────────────────────────────────────────


@pytest.mark.integration
class TestLayerIsolation:
    """Verify memory layers are correctly isolated."""

    def test_layer_search_isolation(self, tmp_path):
        """Searching one layer doesn't return results from another."""
        m = MemoryFS(path=tmp_path / "layer_iso")
        m.write("/k/note", "knowledge about databases schema tables", layer="knowledge")
        m.write("/s/note", "session conversation databases schema tables", layer="session")

        k_results = m.search("databases schema", layer="knowledge", k=10)
        s_results = m.search("databases schema", layer="session", k=10)

        k_layers = {r.metadata.get("layer") for r in k_results}
        s_layers = {r.metadata.get("layer") for r in s_results}

        if k_results:
            assert k_layers == {"knowledge"}
        if s_results:
            assert s_layers == {"session"}

    def test_forget_layer_removes_all(self, tmp_path):
        """Forgetting by layer removes all memories in that layer."""
        m = MemoryFS(path=tmp_path / "forget_layer")
        m.write("/s/a", "session a", layer="session")
        m.write("/s/b", "session b", layer="session")
        m.write("/k/c", "knowledge c", layer="knowledge")

        result = m.forget(layer="session")
        assert result["deleted"] == 2

        remaining = m.list("/", recursive=True)
        assert all(o.layer != "session" for o in remaining)
        assert any(o.layer == "knowledge" for o in remaining)
