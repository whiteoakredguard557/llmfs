"""
Result ranker for LLMFS retrieval.

Takes a list of :class:`~llmfs.core.memory_object.SearchResult` objects from
one or more retrieval sources and applies:

1. **Score fusion** — combines multiple signals (semantic similarity, recency,
   tag overlap, graph proximity) using a weighted sum via Reciprocal Rank
   Fusion (RRF).
2. **Diversity filtering** — greedy max-marginal-relevance (MMR)-style
   deduplication so that results covering similar content are spread out.

Both steps are optional and independently configurable.

Example::

    from llmfs.retrieval.ranker import Ranker, RankConfig
    from llmfs.core.memory_object import SearchResult

    config = RankConfig(recency_weight=0.2, diversity_lambda=0.5)
    ranker = Ranker(config)
    ranked = ranker.rank(results, query="authentication bug", top_k=5)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from llmfs.core.memory_object import SearchResult

__all__ = ["RankConfig", "Ranker"]

logger = logging.getLogger(__name__)

# Constant added to each rank to avoid division by zero in RRF
_RRF_K = 60


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class RankConfig:
    """Tuning parameters for the :class:`Ranker`.

    Attributes:
        semantic_weight: Weight applied to the raw semantic similarity score.
        recency_weight: Weight applied to the recency score (0 = ignore time).
        tag_weight: Weight applied to tag-overlap bonus.
        graph_weight: Weight for graph-proximity signals (0 = off).
        diversity_lambda: MMR trade-off [0, 1].  1.0 = pure relevance,
            0.0 = pure diversity.  Set to 1.0 to disable diversity.
        recency_half_life_hours: Half-life for the exponential recency decay.
            Memories older than this many hours receive ~50% of their full
            recency bonus.
    """

    semantic_weight: float = 0.7
    recency_weight: float = 0.15
    tag_weight: float = 0.1
    graph_weight: float = 0.05
    diversity_lambda: float = 0.7
    recency_half_life_hours: float = 168.0  # 1 week

    def __post_init__(self) -> None:
        for attr in ("semantic_weight", "recency_weight", "tag_weight", "graph_weight"):
            v = getattr(self, attr)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1], got {v}")
        if not 0.0 <= self.diversity_lambda <= 1.0:
            raise ValueError(f"diversity_lambda must be in [0, 1], got {self.diversity_lambda}")
        if self.recency_half_life_hours <= 0:
            raise ValueError("recency_half_life_hours must be > 0")


# ── Internal scored result ────────────────────────────────────────────────────


@dataclass
class _ScoredResult:
    result: SearchResult
    final_score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)


# ── Ranker ────────────────────────────────────────────────────────────────────


class Ranker:
    """Re-ranks search results by fusing multiple relevance signals.

    Args:
        config: :class:`RankConfig` instance.  Defaults to balanced weights.

    Example::

        ranker = Ranker()
        ranked = ranker.rank(raw_results, query="JWT bug", top_k=5)
    """

    def __init__(self, config: RankConfig | None = None) -> None:
        self._cfg = config or RankConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def rank(
        self,
        results: list[SearchResult],
        *,
        query: str = "",
        query_tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rank and optionally deduplicate *results*.

        Args:
            results: Raw search results (order does not matter).
            query: The original query string (used for tag matching).
            query_tags: Tags associated with the query (boosts tag-matched
                results).
            top_k: If provided, return at most this many results.

        Returns:
            Sorted list of :class:`~llmfs.core.memory_object.SearchResult`,
            highest scoring first, with updated ``score`` values.
        """
        if not results:
            return []

        query_tags = query_tags or []

        # 1. Compute component scores for each result
        scored = [
            self._score_result(r, query_tags=query_tags)
            for r in results
        ]

        # 2. Apply RRF over the semantic ranking as the primary signal
        scored = self._apply_rrf(scored)

        # 3. Sort by final score descending
        scored.sort(key=lambda s: -s.final_score)

        # 4. Diversity filtering (MMR-style, path-based)
        if self._cfg.diversity_lambda < 1.0:
            scored = self._diversify(scored)

        output = [s.result for s in scored]
        if top_k is not None:
            output = output[:top_k]
        return output

    def fuse(
        self,
        *result_lists: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Fuse multiple ranked result lists via Reciprocal Rank Fusion.

        Each list contributes equally.  Results appearing in multiple lists
        receive a score boost from each occurrence.

        Args:
            *result_lists: Variable number of result lists to merge.
            top_k: If provided, return at most this many results.

        Returns:
            Fused and ranked :class:`~llmfs.core.memory_object.SearchResult`
            list.
        """
        rrf_scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                path = result.path
                rrf_scores[path] = rrf_scores.get(path, 0.0) + 1.0 / (_RRF_K + rank + 1)
                # Keep the highest-scoring version of a duplicate path
                if path not in best_result or result.score > best_result[path].score:
                    best_result[path] = result

        # Build output with fused scores
        fused: list[SearchResult] = []
        for path, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            r = best_result[path]
            fused.append(SearchResult(
                path=r.path,
                content=r.content,
                score=rrf_score,
                metadata=r.metadata,
                tags=r.tags,
                chunk_text=r.chunk_text,
            ))

        if top_k is not None:
            fused = fused[:top_k]
        return fused

    # ── Internal scoring ──────────────────────────────────────────────────────

    def _score_result(
        self,
        result: SearchResult,
        query_tags: list[str],
    ) -> _ScoredResult:
        """Compute a weighted composite score for a single result."""
        cfg = self._cfg
        components: dict[str, float] = {}

        # Semantic similarity (already in [0, 1])
        sem = float(result.score)
        components["semantic"] = sem

        # Recency score: exponential decay based on modified_at
        rec = self._recency_score(result.metadata)
        components["recency"] = rec

        # Tag overlap score
        tag = self._tag_score(result.tags, query_tags)
        components["tag"] = tag

        final = (
            cfg.semantic_weight * sem
            + cfg.recency_weight * rec
            + cfg.tag_weight * tag
        )
        components["final_before_rrf"] = final

        # Write back the composite score to the result copy
        updated = SearchResult(
            path=result.path,
            content=result.content,
            score=final,
            metadata=result.metadata,
            tags=result.tags,
            chunk_text=result.chunk_text,
        )
        return _ScoredResult(result=updated, final_score=final, component_scores=components)

    def _recency_score(self, metadata: dict[str, Any]) -> float:
        """Return a recency score in [0, 1] based on ``modified_at``."""
        ts = metadata.get("modified_at") or metadata.get("created_at")
        if not ts:
            return 0.5  # unknown age → neutral
        try:
            then = datetime.fromisoformat(ts)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hours_ago = (now - then).total_seconds() / 3600.0
            hl = self._cfg.recency_half_life_hours
            return math.exp(-math.log(2) * hours_ago / hl)
        except (ValueError, OverflowError):
            return 0.5

    @staticmethod
    def _tag_score(result_tags: list[str], query_tags: list[str]) -> float:
        """Jaccard similarity between result tags and query tags."""
        if not query_tags:
            return 0.0
        r_set = set(result_tags)
        q_set = set(query_tags)
        union = r_set | q_set
        if not union:
            return 0.0
        return len(r_set & q_set) / len(union)

    def _apply_rrf(self, scored: list[_ScoredResult]) -> list[_ScoredResult]:
        """Re-score using Reciprocal Rank Fusion over the current ordering."""
        # Sort by current composite score to assign ranks
        sorted_by_score = sorted(scored, key=lambda s: -s.final_score)
        rrf_map: dict[str, float] = {}
        for rank, s in enumerate(sorted_by_score):
            path = s.result.path
            rrf_map[path] = rrf_map.get(path, 0.0) + 1.0 / (_RRF_K + rank + 1)

        for s in scored:
            s.final_score = rrf_map.get(s.result.path, 0.0)
            s.result = SearchResult(
                path=s.result.path,
                content=s.result.content,
                score=s.final_score,
                metadata=s.result.metadata,
                tags=s.result.tags,
                chunk_text=s.result.chunk_text,
            )
        return scored

    def _diversify(self, scored: list[_ScoredResult]) -> list[_ScoredResult]:
        """Greedy MMR-style diversity filtering using path prefix similarity.

        Paths sharing the same parent directory are considered similar.  The
        first result is always kept; subsequent results are penalised if they
        share a parent directory with an already-selected result.

        Args:
            scored: Results sorted by descending score.

        Returns:
            Reordered list balancing relevance and path diversity.
        """
        λ = self._cfg.diversity_lambda
        if not scored:
            return scored

        selected: list[_ScoredResult] = [scored[0]]
        selected_prefixes: set[str] = {_parent_prefix(scored[0].result.path)}
        remaining = scored[1:]

        while remaining:
            best_idx = -1
            best_score = -1.0
            for i, candidate in enumerate(remaining):
                prefix = _parent_prefix(candidate.result.path)
                # Diversity penalty: how many selected results share this prefix?
                overlap = sum(1 for sp in selected_prefixes if sp == prefix)
                diversity = 1.0 / (1.0 + overlap)
                mmr_score = λ * candidate.final_score + (1 - λ) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(remaining.pop(best_idx))
            selected_prefixes.add(_parent_prefix(selected[-1].result.path))

        return selected

    def __repr__(self) -> str:
        return f"Ranker(config={self._cfg!r})"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parent_prefix(path: str) -> str:
    """Return the parent directory of a memory path, e.g. ``/a/b`` for ``/a/b/c``.

    For a root-level path like ``/a``, returns ``/``.
    """
    stripped = path.rstrip("/")
    parts = stripped.rsplit("/", 1)
    if len(parts) == 2 and parts[0]:
        return parts[0]
    return "/"
