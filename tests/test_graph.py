"""
Tests for llmfs.graph.memory_graph.MemoryGraph.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llmfs.core.exceptions import MemoryNotFoundError
from llmfs.graph.memory_graph import GraphEdge, MemoryGraph, TraversalResult
from llmfs.storage.metadata_db import MetadataDB


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path: Path) -> MetadataDB:
    return MetadataDB(tmp_path / "meta.db")


def _add_file(db: MetadataDB, path: str, layer: str = "knowledge") -> str:
    """Insert a minimal file row and return its id."""
    import uuid
    fid = str(uuid.uuid4())
    db.insert_file(
        id=fid,
        path=path,
        name=path.rsplit("/", 1)[-1],
        layer=layer,
        size=0,
        created_at="2026-01-01T00:00:00+00:00",
        modified_at="2026-01-01T00:00:00+00:00",
    )
    return fid


@pytest.fixture
def graph(db: MetadataDB) -> MemoryGraph:
    return MemoryGraph(db)


@pytest.fixture
def populated_graph(db: MetadataDB, graph: MemoryGraph):
    """Graph with nodes A→B, A→C, B→D, C→D.

       A → B → D
       ↓       ↑
       C ──────┘
    """
    for path in ["/a", "/b", "/c", "/d"]:
        _add_file(db, path)
    graph.add_edge("/a", "/b", rel_type="related_to", strength=0.9)
    graph.add_edge("/a", "/c", rel_type="follows", strength=0.7)
    graph.add_edge("/b", "/d", rel_type="related_to", strength=0.8)
    graph.add_edge("/c", "/d", rel_type="caused_by", strength=0.6)
    return graph


# ── GraphEdge ─────────────────────────────────────────────────────────────────


class TestGraphEdge:
    def test_to_dict_keys(self):
        edge = GraphEdge(
            id="e1", source_path="/a", target_path="/b",
            rel_type="related_to", strength=0.5
        )
        d = edge.to_dict()
        assert set(d.keys()) == {"id", "source_path", "target_path", "rel_type", "strength", "created_at"}

    def test_strength_in_dict(self):
        edge = GraphEdge(id="x", source_path="/s", target_path="/t", rel_type="follows", strength=0.75)
        assert edge.to_dict()["strength"] == 0.75


# ── TraversalResult ───────────────────────────────────────────────────────────


class TestTraversalResult:
    def test_to_dict(self):
        r = TraversalResult(root="/a", visited=["/a", "/b"], depth_map={"/a": 0, "/b": 1})
        d = r.to_dict()
        assert d["root"] == "/a"
        assert d["visited"] == ["/a", "/b"]
        assert d["depth_map"] == {"/a": 0, "/b": 1}


# ── add_edge ──────────────────────────────────────────────────────────────────


class TestAddEdge:
    def test_returns_graph_edge(self, db, graph):
        _add_file(db, "/x")
        _add_file(db, "/y")
        edge = graph.add_edge("/x", "/y")
        assert isinstance(edge, GraphEdge)
        assert edge.source_path == "/x"
        assert edge.target_path == "/y"
        assert edge.rel_type == "related_to"

    def test_custom_rel_type_and_strength(self, db, graph):
        _add_file(db, "/src")
        _add_file(db, "/dst")
        edge = graph.add_edge("/src", "/dst", rel_type="caused_by", strength=0.3)
        assert edge.rel_type == "caused_by"
        assert edge.strength == pytest.approx(0.3)

    def test_missing_source_raises(self, db, graph):
        _add_file(db, "/dst")
        with pytest.raises(MemoryNotFoundError):
            graph.add_edge("/nonexistent", "/dst")

    def test_missing_target_raises(self, db, graph):
        _add_file(db, "/src")
        with pytest.raises(MemoryNotFoundError):
            graph.add_edge("/src", "/nonexistent")

    def test_invalid_strength_raises(self, db, graph):
        _add_file(db, "/p")
        _add_file(db, "/q")
        with pytest.raises(ValueError, match="strength"):
            graph.add_edge("/p", "/q", strength=1.5)

    def test_strength_zero_valid(self, db, graph):
        _add_file(db, "/p")
        _add_file(db, "/q")
        edge = graph.add_edge("/p", "/q", strength=0.0)
        assert edge.strength == pytest.approx(0.0)


# ── remove_edge ───────────────────────────────────────────────────────────────


class TestRemoveEdge:
    def test_remove_deletes_edge(self, db, graph):
        _add_file(db, "/r1")
        _add_file(db, "/r2")
        edge = graph.add_edge("/r1", "/r2")
        graph.remove_edge(edge.id)
        # After removal, neighbours should be empty
        neighbours = graph.neighbours("/r1")
        assert all(e.id != edge.id for e in neighbours)


# ── neighbours ────────────────────────────────────────────────────────────────


class TestNeighbours:
    def test_outgoing_neighbours(self, populated_graph):
        edges = populated_graph.neighbours("/a", direction="outgoing")
        target_paths = {e.target_path for e in edges}
        assert "/b" in target_paths
        assert "/c" in target_paths

    def test_sorted_by_strength_desc(self, populated_graph):
        edges = populated_graph.neighbours("/a")
        strengths = [e.strength for e in edges]
        assert strengths == sorted(strengths, reverse=True)

    def test_filter_by_rel_type(self, populated_graph):
        edges = populated_graph.neighbours("/a", rel_type="related_to")
        assert all(e.rel_type == "related_to" for e in edges)

    def test_filter_by_min_strength(self, populated_graph):
        edges = populated_graph.neighbours("/a", min_strength=0.8)
        assert all(e.strength >= 0.8 for e in edges)

    def test_incoming_direction(self, db, graph):
        _add_file(db, "/m")
        _add_file(db, "/n")
        graph.add_edge("/m", "/n", strength=0.9)
        # /n should have /m as incoming neighbour
        incoming = graph.neighbours("/n", direction="incoming")
        assert any(e.source_path == "/m" for e in incoming)

    def test_invalid_direction_raises(self, populated_graph):
        with pytest.raises(ValueError, match="direction"):
            populated_graph.neighbours("/a", direction="sideways")

    def test_missing_path_raises(self, graph):
        with pytest.raises(MemoryNotFoundError):
            graph.neighbours("/missing")


# ── BFS ───────────────────────────────────────────────────────────────────────


class TestBFS:
    def test_bfs_visits_root(self, populated_graph):
        result = populated_graph.bfs("/a", depth=1)
        assert result.root == "/a"
        assert "/a" in result.visited

    def test_bfs_depth_1_no_grandchildren(self, populated_graph):
        result = populated_graph.bfs("/a", depth=1)
        # At depth 1, /b and /c are reachable but /d (depth 2) is not
        assert "/b" in result.visited
        assert "/c" in result.visited
        assert "/d" not in result.visited

    def test_bfs_depth_2_includes_grandchildren(self, populated_graph):
        result = populated_graph.bfs("/a", depth=2)
        assert "/d" in result.visited

    def test_bfs_depth_map_correct(self, populated_graph):
        result = populated_graph.bfs("/a", depth=2)
        assert result.depth_map["/a"] == 0
        assert result.depth_map["/b"] == 1
        assert result.depth_map["/c"] == 1
        assert result.depth_map["/d"] == 2

    def test_bfs_no_duplicates_in_visited(self, populated_graph):
        result = populated_graph.bfs("/a", depth=3)
        assert len(result.visited) == len(set(result.visited))

    def test_bfs_max_nodes_respected(self, populated_graph):
        result = populated_graph.bfs("/a", depth=5, max_nodes=2)
        assert len(result.visited) <= 2

    def test_bfs_depth_0_returns_only_root(self, populated_graph):
        result = populated_graph.bfs("/a", depth=0)
        assert result.visited == ["/a"]

    def test_bfs_missing_start_raises(self, graph):
        with pytest.raises(MemoryNotFoundError):
            graph.bfs("/nonexistent")

    def test_bfs_rel_type_filter(self, populated_graph):
        result = populated_graph.bfs("/a", depth=2, rel_type="related_to")
        # /c reachable only via 'follows', so should be excluded
        assert "/c" not in result.visited

    def test_bfs_returns_traversal_result(self, populated_graph):
        result = populated_graph.bfs("/a", depth=1)
        assert isinstance(result, TraversalResult)


# ── DFS ───────────────────────────────────────────────────────────────────────


class TestDFS:
    def test_dfs_visits_all_reachable(self, populated_graph):
        result = populated_graph.dfs("/a", depth=3)
        assert "/a" in result.visited
        assert "/b" in result.visited
        assert "/c" in result.visited
        assert "/d" in result.visited

    def test_dfs_no_duplicates(self, populated_graph):
        result = populated_graph.dfs("/a", depth=3)
        assert len(result.visited) == len(set(result.visited))

    def test_dfs_depth_0_returns_only_root(self, populated_graph):
        result = populated_graph.dfs("/a", depth=0)
        assert result.visited == ["/a"]

    def test_dfs_missing_start_raises(self, graph):
        with pytest.raises(MemoryNotFoundError):
            graph.dfs("/nonexistent")


# ── shortest_path ─────────────────────────────────────────────────────────────


class TestShortestPath:
    def test_direct_path(self, populated_graph):
        path = populated_graph.shortest_path("/a", "/b")
        assert path == ["/a", "/b"]

    def test_two_hop_path(self, populated_graph):
        path = populated_graph.shortest_path("/a", "/d")
        assert path is not None
        assert path[0] == "/a"
        assert path[-1] == "/d"
        assert len(path) == 3  # /a → /b → /d or /a → /c → /d

    def test_no_path_returns_none(self, db, graph):
        _add_file(db, "/isolated")
        _add_file(db, "/also_isolated")
        result = graph.shortest_path("/isolated", "/also_isolated")
        assert result is None

    def test_same_node_path(self, populated_graph):
        # Shortest path from a node to itself
        path = populated_graph.shortest_path("/a", "/a")
        assert path == ["/a"]

    def test_missing_source_raises(self, graph):
        with pytest.raises(MemoryNotFoundError):
            graph.shortest_path("/missing", "/also_missing")
