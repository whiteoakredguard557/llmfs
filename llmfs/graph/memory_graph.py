"""
MemoryGraph — knowledge-graph layer for LLMFS.

Wraps the relationship tables in :class:`~llmfs.storage.metadata_db.MetadataDB`
with a clean graph API: add/remove edges, query neighbours, and traverse the
graph via breadth-first search (BFS) or depth-first search (DFS).

Relationship types supported (extensible via the *rel_type* argument):

- ``related_to`` — generic semantic connection
- ``follows`` — temporal succession (B happened after A)
- ``caused_by`` — causal link
- ``contradicts`` — conflicting information

All paths returned are **memory paths** (e.g. ``/projects/auth/debug``), not
internal UUIDs.

Example::

    from llmfs import MemoryFS
    from llmfs.graph.memory_graph import MemoryGraph

    mem = MemoryFS()
    graph = MemoryGraph(mem._db)
    graph.add_edge("/a/b", "/a/c", rel_type="related_to", strength=0.9)
    neighbours = graph.neighbours("/a/b")
    traversal  = graph.bfs("/a/b", depth=2)
"""
from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from llmfs.core.exceptions import MemoryNotFoundError, StorageError
from llmfs.storage.metadata_db import MetadataDB

__all__ = ["GraphEdge", "TraversalResult", "MemoryGraph"]

logger = logging.getLogger(__name__)

_VALID_REL_TYPES = frozenset({"related_to", "follows", "caused_by", "contradicts"})


# ── Value objects ─────────────────────────────────────────────────────────────


@dataclass
class GraphEdge:
    """A directed edge in the LLMFS memory graph.

    Attributes:
        id: UUID of this relationship row.
        source_path: Path of the source memory.
        target_path: Path of the target memory.
        rel_type: Relationship type (``related_to``, ``follows``, etc.).
        strength: Numeric weight in [0.0, 1.0]; higher = stronger link.
        created_at: ISO-8601 UTC creation timestamp.
    """

    id: str
    source_path: str
    target_path: str
    rel_type: str
    strength: float
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "rel_type": self.rel_type,
            "strength": self.strength,
            "created_at": self.created_at,
        }


@dataclass
class TraversalResult:
    """The outcome of a BFS or DFS graph traversal.

    Attributes:
        root: The starting memory path.
        visited: Ordered list of memory paths reached (root-first).
        edges: All edges traversed, in discovery order.
        depth_map: Maps each visited path to its BFS depth from the root.
    """

    root: str
    visited: list[str] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    depth_map: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "visited": self.visited,
            "edges": [e.to_dict() for e in self.edges],
            "depth_map": self.depth_map,
        }


# ── Main class ────────────────────────────────────────────────────────────────


class MemoryGraph:
    """Relationship graph over LLMFS memories.

    Wraps :class:`~llmfs.storage.metadata_db.MetadataDB` relationship tables
    with a convenient graph API.  All operations accept and return **memory
    paths** rather than internal UUIDs.

    Args:
        db: An initialised :class:`~llmfs.storage.metadata_db.MetadataDB`
            instance.

    Example::

        graph = MemoryGraph(db)
        graph.add_edge("/a", "/b", rel_type="related_to", strength=0.8)
        result = graph.bfs("/a", depth=2)
    """

    def __init__(self, db: MetadataDB) -> None:
        self._db = db

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> dict[str, Any]:
        """Return the DB row for *path*, raising :class:`MemoryNotFoundError` if absent."""
        row = self._db.get_file(path)
        if not row:
            raise MemoryNotFoundError(path)
        return row

    def _edge_from_row(
        self,
        row: dict[str, Any],
        source_path: str,
        target_path: str,
    ) -> GraphEdge:
        return GraphEdge(
            id=row["id"],
            source_path=source_path,
            target_path=target_path,
            rel_type=row["type"],
            strength=float(row["strength"]),
            created_at=row.get("created_at", ""),
        )

    # ── Mutating operations ───────────────────────────────────────────────────

    def add_edge(
        self,
        source_path: str,
        target_path: str,
        *,
        rel_type: str = "related_to",
        strength: float = 0.8,
    ) -> GraphEdge:
        """Add (or replace) a directed relationship edge.

        Args:
            source_path: Memory path for the source node.
            target_path: Memory path for the target node.
            rel_type: Relationship type; must be one of ``related_to``,
                ``follows``, ``caused_by``, ``contradicts``, or any custom
                string.
            strength: Edge weight in [0.0, 1.0].

        Returns:
            The newly created :class:`GraphEdge`.

        Raises:
            MemoryNotFoundError: If either path does not exist.
            ValueError: If *strength* is outside [0, 1].
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {strength}")

        src_row = self._resolve_path(source_path)
        tgt_row = self._resolve_path(target_path)

        rel_id = str(uuid.uuid4())
        try:
            self._db.insert_relationship(
                id=rel_id,
                source_id=src_row["id"],
                target_id=tgt_row["id"],
                rel_type=rel_type,
                strength=strength,
            )
        except StorageError as exc:
            raise StorageError(f"Failed to add edge {source_path!r} -> {target_path!r}: {exc}") from exc

        logger.debug(
            "add_edge: %s -[%s]-> %s (%.2f)", source_path, rel_type, target_path, strength
        )
        return GraphEdge(
            id=rel_id,
            source_path=source_path,
            target_path=target_path,
            rel_type=rel_type,
            strength=strength,
        )

    def remove_edge(self, edge_id: str) -> None:
        """Delete a relationship by its UUID.

        Args:
            edge_id: The ``id`` field of the :class:`GraphEdge` to remove.

        Raises:
            StorageError: On database error.
        """
        self._db.delete_relationship(edge_id)
        logger.debug("remove_edge: %s", edge_id)

    # ── Query operations ──────────────────────────────────────────────────────

    def neighbours(
        self,
        path: str,
        *,
        rel_type: str | None = None,
        min_strength: float = 0.0,
        direction: str = "outgoing",
    ) -> list[GraphEdge]:
        """Return the direct neighbours of *path*.

        Args:
            path: Source memory path.
            rel_type: Filter to a specific relationship type.  ``None`` returns
                all types.
            min_strength: Only include edges with ``strength >= min_strength``.
            direction: ``"outgoing"`` (default) returns edges *from* this node;
                ``"incoming"`` returns edges *to* this node; ``"both"``
                returns all.

        Returns:
            List of :class:`GraphEdge` objects sorted by descending strength.

        Raises:
            MemoryNotFoundError: If *path* does not exist.
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"direction must be 'outgoing', 'incoming', or 'both', got {direction!r}")

        src_row = self._resolve_path(path)
        file_id = src_row["id"]

        edges: list[GraphEdge] = []

        if direction in ("outgoing", "both"):
            for row in self._db.get_relationships(file_id):
                tgt = self._db.get_file_by_id(row["target_id"])
                if tgt is None:
                    continue
                if rel_type and row["type"] != rel_type:
                    continue
                if float(row["strength"]) < min_strength:
                    continue
                edges.append(self._edge_from_row(row, path, tgt["path"]))

        if direction in ("incoming", "both"):
            for row in self._db.get_incoming_relationships(file_id):
                src = self._db.get_file_by_id(row["source_id"])
                if src is None:
                    continue
                if rel_type and row["type"] != rel_type:
                    continue
                if float(row["strength"]) < min_strength:
                    continue
                edges.append(self._edge_from_row(row, src["path"], path))

        edges.sort(key=lambda e: -e.strength)
        return edges

    def get_edges(self, path: str) -> list[GraphEdge]:
        """Return all outgoing edges from *path*.

        Convenience alias for :meth:`neighbours` with default parameters.
        """
        return self.neighbours(path)

    # ── Traversal ─────────────────────────────────────────────────────────────

    def bfs(
        self,
        start: str,
        *,
        depth: int = 2,
        rel_type: str | None = None,
        min_strength: float = 0.0,
        max_nodes: int = 100,
    ) -> TraversalResult:
        """Breadth-first traversal of the memory graph.

        Starts from *start* and expands outgoing edges layer by layer up to
        *depth* hops.

        Args:
            start: Root memory path for the traversal.
            depth: Maximum number of hops from the root (inclusive).
            rel_type: Filter to a specific relationship type.
            min_strength: Only follow edges with ``strength >= min_strength``.
            max_nodes: Hard cap on the number of nodes visited (safety valve).

        Returns:
            :class:`TraversalResult` with all visited paths, edges, and depth
            information.

        Raises:
            MemoryNotFoundError: If *start* does not exist.
            ValueError: If *depth* < 0 or *max_nodes* < 1.
        """
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")

        result = TraversalResult(root=start)
        self._resolve_path(start)  # validate existence

        queue: deque[tuple[str, int]] = deque([(start, 0)])
        visited: set[str] = {start}
        result.visited.append(start)
        result.depth_map[start] = 0

        while queue and len(result.visited) < max_nodes:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            try:
                edges = self.neighbours(
                    current,
                    rel_type=rel_type,
                    min_strength=min_strength,
                    direction="outgoing",
                )
            except MemoryNotFoundError:
                # Node was deleted during traversal
                continue

            for edge in edges:
                result.edges.append(edge)
                neighbour = edge.target_path
                if neighbour not in visited:
                    visited.add(neighbour)
                    result.visited.append(neighbour)
                    result.depth_map[neighbour] = current_depth + 1
                    queue.append((neighbour, current_depth + 1))
                    if len(result.visited) >= max_nodes:
                        break

        logger.debug(
            "bfs from %s: visited %d nodes, %d edges",
            start, len(result.visited), len(result.edges),
        )
        return result

    def dfs(
        self,
        start: str,
        *,
        depth: int = 2,
        rel_type: str | None = None,
        min_strength: float = 0.0,
        max_nodes: int = 100,
    ) -> TraversalResult:
        """Depth-first traversal of the memory graph.

        Args:
            start: Root memory path for the traversal.
            depth: Maximum number of hops from the root.
            rel_type: Filter to a specific relationship type.
            min_strength: Only follow edges with ``strength >= min_strength``.
            max_nodes: Hard cap on the number of nodes visited.

        Returns:
            :class:`TraversalResult` with all visited paths, edges, and depth
            information.

        Raises:
            MemoryNotFoundError: If *start* does not exist.
        """
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")

        result = TraversalResult(root=start)
        self._resolve_path(start)  # validate existence

        visited: set[str] = set()

        def _dfs(node: str, current_depth: int) -> None:
            if node in visited or len(result.visited) >= max_nodes:
                return
            visited.add(node)
            if node not in result.depth_map:
                result.depth_map[node] = current_depth
            result.visited.append(node)

            if current_depth >= depth:
                return
            try:
                edges = self.neighbours(
                    node,
                    rel_type=rel_type,
                    min_strength=min_strength,
                    direction="outgoing",
                )
            except MemoryNotFoundError:
                return

            for edge in edges:
                result.edges.append(edge)
                _dfs(edge.target_path, current_depth + 1)

        _dfs(start, 0)
        logger.debug(
            "dfs from %s: visited %d nodes, %d edges",
            start, len(result.visited), len(result.edges),
        )
        return result

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find the shortest path from *source* to *target* via BFS.

        Args:
            source: Starting memory path.
            target: Destination memory path.
            max_depth: Maximum hops to search.

        Returns:
            Ordered list of paths from *source* to *target* (inclusive), or
            ``None`` if no path exists within *max_depth* hops.

        Raises:
            MemoryNotFoundError: If *source* does not exist.
        """
        self._resolve_path(source)

        # Parent tracking for path reconstruction
        parent: dict[str, str | None] = {source: None}
        queue: deque[tuple[str, int]] = deque([(source, 0)])

        while queue:
            current, d = queue.popleft()
            if current == target:
                # Reconstruct path
                path: list[str] = []
                node: str | None = target
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            if d >= max_depth:
                continue
            try:
                edges = self.neighbours(current, direction="outgoing")
            except MemoryNotFoundError:
                continue
            for edge in edges:
                nbr = edge.target_path
                if nbr not in parent:
                    parent[nbr] = current
                    queue.append((nbr, d + 1))

        return None

    def __repr__(self) -> str:
        return f"MemoryGraph(db={self._db!r})"
