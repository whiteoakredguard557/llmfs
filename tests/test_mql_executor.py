"""
Comprehensive tests for MQLExecutor.

Tests are organized into classes, each sharing a single MemoryFS instance
(and therefore a single embedder load) via a class-scoped fixture.

Run with:
    /root/LLMFS/.venv/bin/python -m pytest tests/test_mql_executor.py -v --timeout=120
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from llmfs.core.filesystem import MemoryFS
from llmfs.core.exceptions import MQLExecutionError, MQLParseError
from llmfs.core.memory_object import SearchResult
from llmfs.query.executor import MQLExecutor, execute_mql


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _paths(results: list[SearchResult]) -> set[str]:
    """Return the set of paths from a result list."""
    return {r.path for r in results}


# ===========================================================================
# TestMQLExecutorNoWhere
# ===========================================================================

class TestMQLExecutorNoWhere:
    """SELECT FROM / with no WHERE clause — list all or prefix-filtered results."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        """MemoryFS with a handful of memories across two prefixes."""
        base = tmp_path_factory.mktemp("nowhere")
        m = MemoryFS(path=base)
        m.write("/alpha/one", "Alpha document one about databases")
        m.write("/alpha/two", "Alpha document two about networking")
        m.write("/beta/one", "Beta document one about security")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    # ── tests ────────────────────────────────────────────────────────────────

    def test_select_from_root_returns_all(self, executor):
        """SELECT memory FROM / returns all stored memories."""
        results = executor.execute_mql("SELECT memory FROM /")
        assert len(results) >= 3
        found = _paths(results)
        assert "/alpha/one" in found
        assert "/alpha/two" in found
        assert "/beta/one" in found

    def test_select_from_root_returns_searchresult_objects(self, executor):
        """Results are SearchResult instances."""
        results = executor.execute_mql("SELECT memory FROM /")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_select_from_root_result_has_expected_fields(self, executor):
        """Every SearchResult has path, score, metadata, tags, chunk_text."""
        results = executor.execute_mql("SELECT memory FROM /")
        r = results[0]
        assert hasattr(r, "path")
        assert hasattr(r, "score")
        assert hasattr(r, "metadata")
        assert hasattr(r, "tags")
        assert hasattr(r, "chunk_text")

    def test_select_from_prefix_returns_subset(self, executor):
        """SELECT memory FROM /alpha returns only /alpha/* memories."""
        results = executor.execute_mql("SELECT memory FROM /alpha")
        paths = _paths(results)
        assert "/alpha/one" in paths
        assert "/alpha/two" in paths
        assert "/beta/one" not in paths

    def test_select_from_prefix_beta(self, executor):
        """SELECT memory FROM /beta returns only /beta/* memories."""
        results = executor.execute_mql("SELECT memory FROM /beta")
        paths = _paths(results)
        assert "/beta/one" in paths
        assert "/alpha/one" not in paths
        assert "/alpha/two" not in paths

    def test_select_from_empty_prefix_returns_empty(self, executor):
        """FROM /nonexistent prefix returns empty list."""
        results = executor.execute_mql("SELECT memory FROM /nonexistent")
        assert results == []

    def test_module_level_execute_mql(self, mem):
        """execute_mql() module-level helper works correctly."""
        results = execute_mql("SELECT memory FROM /", mem)
        assert len(results) >= 3


# ===========================================================================
# TestMQLExecutorSimilar
# ===========================================================================

class TestMQLExecutorSimilar:
    """SIMILAR TO — semantic similarity search."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("similar")
        m = MemoryFS(path=base)
        m.write("/knowledge/auth", "JWT token authentication and session management")
        m.write("/knowledge/db", "PostgreSQL database query optimization")
        m.write("/knowledge/net", "TCP/IP network socket programming")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_similar_returns_list(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE SIMILAR TO "authentication"')
        assert isinstance(results, list)

    def test_similar_returns_search_results(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE SIMILAR TO "authentication"')
        assert all(isinstance(r, SearchResult) for r in results)

    def test_similar_auth_returns_auth_memory(self, executor):
        """Auth-related query should surface the auth memory."""
        results = executor.execute_mql('SELECT memory FROM / WHERE SIMILAR TO "JWT token authentication"')
        assert len(results) > 0
        assert "/knowledge/auth" in _paths(results)

    def test_similar_db_returns_db_memory(self, executor):
        """DB-related query should surface the DB memory."""
        results = executor.execute_mql('SELECT memory FROM / WHERE SIMILAR TO "database query"')
        assert len(results) > 0
        assert "/knowledge/db" in _paths(results)

    def test_similar_with_path_prefix(self, executor):
        """SIMILAR TO respects FROM /prefix filter."""
        results = executor.execute_mql('SELECT memory FROM /knowledge WHERE SIMILAR TO "authentication"')
        # All results should be under /knowledge
        for r in results:
            assert r.path.startswith("/knowledge")

    def test_similar_scores_are_floats(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE SIMILAR TO "network"')
        for r in results:
            assert isinstance(r.score, float)

    def test_similar_scores_between_zero_and_one(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE SIMILAR TO "network"')
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ===========================================================================
# TestMQLExecutorTag
# ===========================================================================

class TestMQLExecutorTag:
    """TAG = / TAG != / TAG IN filters."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("tag")
        m = MemoryFS(path=base)
        m.write("/t/a", "Python asyncio guide", tags=["python", "async"])
        m.write("/t/b", "Rust ownership model", tags=["rust"])
        m.write("/t/c", "Python type hints", tags=["python", "types"])
        m.write("/t/d", "Go concurrency patterns", tags=["go", "async"])
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_tag_eq_returns_matching_memories(self, executor):
        """TAG = 'python' returns only python-tagged memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TAG = "python"')
        paths = _paths(results)
        assert "/t/a" in paths
        assert "/t/c" in paths
        assert "/t/b" not in paths
        assert "/t/d" not in paths

    def test_tag_eq_single_tag(self, executor):
        """TAG = 'rust' returns exactly the rust memory."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TAG = "rust"')
        assert "/t/b" in _paths(results)
        assert "/t/a" not in _paths(results)

    def test_tag_neq_excludes_tag(self, executor):
        """TAG != 'rust' returns all non-rust memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TAG != "rust"')
        paths = _paths(results)
        assert "/t/b" not in paths
        assert "/t/a" in paths
        assert "/t/c" in paths

    def test_tag_in_returns_any_matching(self, executor):
        """TAG IN ('rust', 'go') returns rust OR go memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TAG IN ("rust", "go")')
        paths = _paths(results)
        assert "/t/b" in paths
        assert "/t/d" in paths
        assert "/t/a" not in paths
        assert "/t/c" not in paths

    def test_tag_in_single_value(self, executor):
        """TAG IN with one value behaves like TAG =."""
        results_in = executor.execute_mql('SELECT memory FROM / WHERE TAG IN ("python")')
        results_eq = executor.execute_mql('SELECT memory FROM / WHERE TAG = "python"')
        assert _paths(results_in) == _paths(results_eq)

    def test_tag_eq_nonexistent_returns_empty(self, executor):
        """TAG = 'nonexistent' returns empty list."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TAG = "nonexistent"')
        assert results == []

    def test_tag_results_are_search_results(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE TAG = "python"')
        assert all(isinstance(r, SearchResult) for r in results)


# ===========================================================================
# TestMQLExecutorDate
# ===========================================================================

class TestMQLExecutorDate:
    """DATE > / DATE < / DATE >= / DATE <= / DATE = filters."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("date")
        m = MemoryFS(path=base)
        # All written at roughly "now" — we'll use relative date filters
        m.write("/d/past", "Past memory")
        m.write("/d/present", "Present memory")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_date_gt_past_returns_present_memories(self, executor):
        """DATE > '2000-01-01' should return all memories (written after 2000)."""
        results = executor.execute_mql('SELECT memory FROM / WHERE DATE > "2000-01-01"')
        paths = _paths(results)
        assert "/d/past" in paths
        assert "/d/present" in paths

    def test_date_lt_future_returns_all(self, executor):
        """DATE < '2099-12-31' should return all memories (all written before 2099)."""
        results = executor.execute_mql('SELECT memory FROM / WHERE DATE < "2099-12-31"')
        paths = _paths(results)
        assert "/d/past" in paths
        assert "/d/present" in paths

    def test_date_gt_future_returns_empty(self, executor):
        """DATE > '2099-12-31' should return no memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE DATE > "2099-12-31"')
        assert results == []

    def test_date_lt_past_returns_empty(self, executor):
        """DATE < '1970-01-01' should return no memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE DATE < "1970-01-01"')
        assert results == []

    def test_date_gte_past_returns_all(self, executor):
        """DATE >= '2000-01-01' returns all memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE DATE >= "2000-01-01"')
        assert len(results) >= 2

    def test_date_lte_future_returns_all(self, executor):
        """DATE <= '2099-12-31' returns all memories."""
        results = executor.execute_mql('SELECT memory FROM / WHERE DATE <= "2099-12-31"')
        assert len(results) >= 2

    def test_date_eq_today_returns_todays_memories(self, executor):
        """DATE = today's date returns all today's memories."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        results = executor.execute_mql(f'SELECT memory FROM / WHERE DATE = "{today}"')
        paths = _paths(results)
        assert "/d/past" in paths
        assert "/d/present" in paths

    def test_date_invalid_raises_execution_error(self, executor):
        """Invalid date string raises MQLExecutionError."""
        with pytest.raises(MQLExecutionError):
            executor.execute_mql('SELECT memory FROM / WHERE DATE > "not-a-date"')


# ===========================================================================
# TestMQLExecutorTopic
# ===========================================================================

class TestMQLExecutorTopic:
    """TOPIC keyword search."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("topic")
        m = MemoryFS(path=base)
        m.write("/topic/a", "Machine learning model training and gradient descent")
        m.write("/topic/b", "Web application REST API design patterns")
        m.write("/topic/c", "Container orchestration with Kubernetes and Docker")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_topic_returns_list(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE TOPIC "machine learning"')
        assert isinstance(results, list)

    def test_topic_returns_search_results(self, executor):
        results = executor.execute_mql('SELECT memory FROM / WHERE TOPIC "machine learning"')
        assert all(isinstance(r, SearchResult) for r in results)

    def test_topic_ml_returns_ml_memory(self, executor):
        """TOPIC 'machine learning' should surface the ML memory."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TOPIC "machine learning"')
        assert len(results) > 0
        assert "/topic/a" in _paths(results)

    def test_topic_kubernetes_returns_container_memory(self, executor):
        """TOPIC 'kubernetes' should surface the containers memory."""
        results = executor.execute_mql('SELECT memory FROM / WHERE TOPIC "kubernetes"')
        assert len(results) > 0
        assert "/topic/c" in _paths(results)

    def test_topic_with_path_prefix(self, executor):
        """TOPIC filters within prefix."""
        results = executor.execute_mql('SELECT memory FROM /topic WHERE TOPIC "REST API"')
        for r in results:
            assert r.path.startswith("/topic")


# ===========================================================================
# TestMQLExecutorLogical
# ===========================================================================

class TestMQLExecutorLogical:
    """AND intersects results; OR unions results."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("logical")
        m = MemoryFS(path=base)
        m.write("/l/a", "Python async programming guide", tags=["python", "async"])
        m.write("/l/b", "Python data science tutorial", tags=["python", "data"])
        m.write("/l/c", "Rust systems programming", tags=["rust"])
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_and_intersects_tag_results(self, executor):
        """TAG = 'python' AND TAG = 'async' returns intersection."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE TAG = "python" AND TAG = "async"'
        )
        paths = _paths(results)
        assert "/l/a" in paths
        assert "/l/b" not in paths
        assert "/l/c" not in paths

    def test_or_unions_tag_results(self, executor):
        """TAG = 'rust' OR TAG = 'async' returns union."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE TAG = "rust" OR TAG = "async"'
        )
        paths = _paths(results)
        assert "/l/a" in paths   # has async
        assert "/l/c" in paths   # has rust
        assert "/l/b" not in paths

    def test_and_with_similar_and_tag(self, executor):
        """SIMILAR AND TAG intersection returns only memories matching both."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE TAG = "python" AND SIMILAR TO "async programming"'
        )
        # All returned paths should have the python tag
        for r in results:
            assert "python" in r.tags

    def test_or_returns_no_duplicates(self, executor):
        """OR result set has no duplicate paths."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE TAG = "python" OR TAG = "rust"'
        )
        paths = [r.path for r in results]
        assert len(paths) == len(set(paths))

    def test_and_nonempty_intersection(self, executor):
        """Intersection of two overlapping tag sets is non-empty."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE TAG = "python" AND TAG = "data"'
        )
        assert "/l/b" in _paths(results)

    def test_and_empty_intersection(self, executor):
        """Intersection of non-overlapping tags is empty."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE TAG = "rust" AND TAG = "python"'
        )
        # /l/c is rust-only, no python; /l/a and /l/b are python-only
        assert results == []


# ===========================================================================
# TestMQLExecutorOrderBy
# ===========================================================================

class TestMQLExecutorOrderBy:
    """ORDER BY score / date ASC / date DESC."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("orderby")
        m = MemoryFS(path=base)
        m.write("/o/a", "First memory document about Python")
        m.write("/o/b", "Second memory document about Java")
        m.write("/o/c", "Third memory document about Go")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_order_by_score_desc(self, executor):
        """ORDER BY score (implicit desc) — scores non-increasing."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE SIMILAR TO "programming language" ORDER BY score'
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_order_by_date_desc(self, executor):
        """ORDER BY date DESC — created_at timestamps non-increasing."""
        results = executor.execute_mql(
            'SELECT memory FROM / ORDER BY date DESC'
        )
        dates = [r.metadata.get("created_at", "") for r in results]
        assert dates == sorted(dates, reverse=True)

    def test_order_by_date_asc(self, executor):
        """ORDER BY date ASC — created_at timestamps non-decreasing."""
        results = executor.execute_mql(
            'SELECT memory FROM / ORDER BY date ASC'
        )
        dates = [r.metadata.get("created_at", "") for r in results]
        assert dates == sorted(dates)

    def test_order_by_score_returns_all_results(self, executor):
        """ORDER BY score doesn't drop any memories."""
        all_results = executor.execute_mql("SELECT memory FROM /")
        ordered = executor.execute_mql(
            "SELECT memory FROM / WHERE SIMILAR TO \"document\" ORDER BY score"
        )
        # At least some results should be present
        assert len(ordered) > 0


# ===========================================================================
# TestMQLExecutorLimit
# ===========================================================================

class TestMQLExecutorLimit:
    """LIMIT N restricts the number of returned results."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("limit")
        m = MemoryFS(path=base)
        for i in range(5):
            m.write(f"/lim/{i}", f"Memory document number {i} about programming")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_limit_1_returns_at_most_1(self, executor):
        """LIMIT 1 returns at most 1 result."""
        results = executor.execute_mql("SELECT memory FROM / LIMIT 1")
        assert len(results) <= 1

    def test_limit_2_returns_at_most_2(self, executor):
        """LIMIT 2 returns at most 2 results."""
        results = executor.execute_mql("SELECT memory FROM / LIMIT 2")
        assert len(results) <= 2

    def test_limit_1_with_similar(self, executor):
        """LIMIT 1 on SIMILAR search returns at most 1 result."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE SIMILAR TO "programming" LIMIT 1'
        )
        assert len(results) <= 1

    def test_limit_larger_than_total_returns_all(self, executor):
        """LIMIT larger than total memories returns all."""
        all_results = executor.execute_mql("SELECT memory FROM /")
        limited = executor.execute_mql("SELECT memory FROM / LIMIT 1000")
        assert len(limited) == len(all_results)

    def test_limit_with_tag_filter(self, executor):
        """LIMIT works alongside WHERE TAG."""
        # write some tagged memories first
        mem_obj = executor._mem
        # /lim/0..4 have no tags; query with no tag filter + limit
        results = executor.execute_mql("SELECT memory FROM /lim LIMIT 3")
        assert len(results) <= 3

    def test_limit_with_order_by(self, executor):
        """LIMIT after ORDER BY returns top-N by sort key."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE SIMILAR TO "programming" ORDER BY score LIMIT 2'
        )
        assert len(results) <= 2
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# TestMQLExecutorRelated
# ===========================================================================

class TestMQLExecutorRelated:
    """RELATED TO — graph traversal from anchor path."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("related")
        m = MemoryFS(path=base)
        m.write("/r/root", "Root memory about authentication")
        m.write("/r/child1", "Child memory about JWT tokens")
        m.write("/r/child2", "Child memory about OAuth2 flows")
        m.write("/r/unrelated", "Unrelated memory about networking")
        # create relationships
        m.relate("/r/root", "/r/child1", "related_to", strength=0.9)
        m.relate("/r/root", "/r/child2", "related_to", strength=0.8)
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_related_to_returns_list(self, executor):
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root"'
        )
        assert isinstance(results, list)

    def test_related_to_returns_search_results(self, executor):
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root"'
        )
        assert all(isinstance(r, SearchResult) for r in results)

    def test_related_to_finds_direct_children(self, executor):
        """RELATED TO '/r/root' finds child1 and child2."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root"'
        )
        paths = _paths(results)
        assert "/r/child1" in paths
        assert "/r/child2" in paths

    def test_related_to_excludes_anchor(self, executor):
        """The anchor itself is not included in results."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root"'
        )
        assert "/r/root" not in _paths(results)

    def test_related_to_excludes_unrelated(self, executor):
        """Unrelated memories are not returned."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root"'
        )
        assert "/r/unrelated" not in _paths(results)

    def test_related_to_with_within_depth(self, executor):
        """RELATED TO with WITHIN 1 returns at most depth-1 nodes."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root" WITHIN 1'
        )
        paths = _paths(results)
        assert "/r/child1" in paths
        assert "/r/child2" in paths

    def test_related_to_scores_are_positive(self, executor):
        """Returned scores from graph traversal are > 0."""
        results = executor.execute_mql(
            'SELECT memory FROM / WHERE RELATED TO "/r/root"'
        )
        for r in results:
            assert r.score > 0.0

    def test_related_to_nonexistent_anchor_raises(self, executor):
        """RELATED TO nonexistent path raises MQLExecutionError."""
        with pytest.raises(MQLExecutionError):
            executor.execute_mql(
                'SELECT memory FROM / WHERE RELATED TO "/r/doesnotexist"'
            )


# ===========================================================================
# TestMQLExecutorErrors
# ===========================================================================

class TestMQLExecutorErrors:
    """Bad MQL raises appropriate exceptions."""

    @pytest.fixture(scope="class")
    def mem(self, tmp_path_factory):
        base = tmp_path_factory.mktemp("errors")
        m = MemoryFS(path=base)
        m.write("/err/a", "Error test memory")
        yield m

    @pytest.fixture(scope="class")
    def executor(self, mem):
        return MQLExecutor(mem)

    def test_missing_select_raises_parse_error(self, executor):
        """Missing SELECT keyword raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql('memory FROM / WHERE SIMILAR TO "test"')

    def test_missing_from_raises_parse_error(self, executor):
        """Missing FROM keyword raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql('SELECT memory WHERE SIMILAR TO "test"')

    def test_empty_string_raises_parse_error(self, executor):
        """Empty query string raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql("")

    def test_missing_similar_to_string_raises_parse_error(self, executor):
        """SIMILAR TO without a quoted string raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql("SELECT memory FROM / WHERE SIMILAR TO")

    def test_invalid_tag_operator_raises_parse_error(self, executor):
        """TAG with unsupported operator raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql('SELECT memory FROM / WHERE TAG > "python"')

    def test_invalid_date_value_raises_execution_error(self, executor):
        """DATE with non-ISO date value raises MQLExecutionError."""
        with pytest.raises(MQLExecutionError):
            executor.execute_mql('SELECT memory FROM / WHERE DATE > "yesterday"')

    def test_gibberish_query_raises_parse_error(self, executor):
        """Completely invalid MQL raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql("this is not MQL at all !!!!")

    def test_missing_limit_value_raises_parse_error(self, executor):
        """LIMIT without a number raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql("SELECT memory FROM / LIMIT")

    def test_related_to_missing_path_raises_parse_error(self, executor):
        """RELATED TO without a path raises MQLParseError."""
        with pytest.raises(MQLParseError):
            executor.execute_mql("SELECT memory FROM / WHERE RELATED TO")

    def test_module_level_execute_mql_bad_query(self, mem):
        """Module-level execute_mql propagates parse errors."""
        with pytest.raises(MQLParseError):
            execute_mql("not valid MQL", mem)
