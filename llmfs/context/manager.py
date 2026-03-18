"""
ContextManager — infinite context for LLMs via LLMFS.

:class:`ContextManager` implements a virtual-memory model for LLM conversations:

- **RAM** ↔ Active context window (recent turns kept in memory)
- **Disk** ↔ LLMFS (all evicted turns stored at full fidelity)
- **Page eviction** ↔ Lowest-importance turns offloaded to LLMFS
- **Page fault** ↔ LLM calls ``memory_search``/``memory_read``

When the active token count crosses a configurable threshold (default 70% of
``max_tokens``), the manager evicts the lowest-importance turns until the
active window drops to 50%.  Before eviction each turn is passed through
:class:`~llmfs.context.extractor.ArtifactExtractor` to store structured
artifacts separately, and then archived as a full turn under
``/session/{session_id}/turns/{id}``.  After eviction a compact memory index
(~2 k tokens) is rebuilt and made available for system-prompt injection.

Example::

    from llmfs import MemoryFS
    from llmfs.context.manager import ContextManager

    mem = MemoryFS(path="/tmp/test")
    ctx = ContextManager(mem=mem, max_tokens=4000, evict_at=0.70)
    ctx.on_new_turn("user", "Fix the auth bug", tokens=10)
    ctx.on_new_turn("assistant", "Found the issue at line 45", tokens=12)
    print(ctx.get_system_prompt_addon())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from llmfs.context.extractor import ArtifactExtractor
from llmfs.context.importance import ImportanceScorer
from llmfs.context.index_builder import IndexBuilder
from llmfs.core.filesystem import MemoryFS

__all__ = ["TurnRecord", "ContextManager"]

logger = logging.getLogger(__name__)


# ── Turn record ───────────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    """A single conversation turn tracked by the :class:`ContextManager`.

    Attributes:
        id: Unique string identifier (incrementing counter as string).
        role: Either ``"user"`` or ``"assistant"``.
        content: Raw text content of the turn.
        tokens: Estimated token count.
        importance: Score assigned by :class:`~llmfs.context.importance.ImportanceScorer`.
        evicted: Whether this turn has been evicted to LLMFS storage.
        artifact_paths: LLMFS paths written during artifact extraction.
    """

    id: str
    role: str
    content: str
    tokens: int
    importance: float = 0.5
    evicted: bool = False
    artifact_paths: list[str] = field(default_factory=list)


# ── ContextManager ────────────────────────────────────────────────────────────


class ContextManager:
    """Manages an LLM context window with automatic memory eviction.

    Args:
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance used for
            persistent storage.
        max_tokens: Capacity of the context window in tokens.
        evict_at: Fraction of ``max_tokens`` at which eviction is triggered.
        target_after_evict: Fraction of ``max_tokens`` to reach after
            eviction.
        session_id: Session identifier used to namespace LLMFS paths.  If
            ``None``, a random UUID hex is generated.
        scorer: :class:`~llmfs.context.importance.ImportanceScorer` instance.
            Defaults to one with default weights.
        extractor: :class:`~llmfs.context.extractor.ArtifactExtractor`.
            Defaults to one with ``layer="session"``.
        index_builder: :class:`~llmfs.context.index_builder.IndexBuilder`.
            Defaults to one with ``max_entries=50``.

    Example::

        ctx = ContextManager(mem=mem, max_tokens=128_000)
        ctx.on_new_turn("user", "Hello!", tokens=5)
        addon = ctx.get_system_prompt_addon()  # inject into system prompt
    """

    def __init__(
        self,
        mem: MemoryFS,
        *,
        max_tokens: int = 128_000,
        evict_at: float = 0.70,
        target_after_evict: float = 0.50,
        session_id: str | None = None,
        scorer: ImportanceScorer | None = None,
        extractor: ArtifactExtractor | None = None,
        index_builder: IndexBuilder | None = None,
    ) -> None:
        self._mem = mem
        self._max_tokens = max_tokens
        self._evict_threshold = int(max_tokens * evict_at)
        self._target_tokens = int(max_tokens * target_after_evict)
        self._session_id: str = session_id or uuid4().hex
        self._scorer = scorer or ImportanceScorer()
        self._extractor = extractor or ArtifactExtractor()
        self._index_builder = index_builder or IndexBuilder()

        self._active_turns: list[TurnRecord] = []
        self._turn_counter: int = 0
        self._total_tokens: int = 0
        self._memory_index: str = ""

        # Stats
        self._evicted_count: int = 0
        self._eviction_rounds: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        """Read-only session identifier."""
        return self._session_id

    def on_new_turn(
        self,
        role: str,
        content: str,
        tokens: int,
    ) -> None:
        """Record a new conversation turn and evict if necessary.

        Args:
            role: ``"user"`` or ``"assistant"``.
            content: Turn text content.
            tokens: Estimated token count for this turn.
        """
        self._turn_counter += 1
        turn_id = str(self._turn_counter)

        # Score importance before adding to active list (for recency context)
        importance = self._scorer.score(
            content,
            role=role,
            turn_index=len(self._active_turns),
            total_turns=len(self._active_turns) + 1,
        )

        turn = TurnRecord(
            id=turn_id,
            role=role,
            content=content,
            tokens=tokens,
            importance=importance,
        )
        self._active_turns.append(turn)
        self._total_tokens += tokens

        logger.debug(
            "on_new_turn: id=%s role=%s tokens=%d importance=%.2f total=%d",
            turn_id, role, tokens, importance, self._total_tokens,
        )

        # Evict if over threshold
        if self._total_tokens >= self._evict_threshold:
            self._evict()
            self._rebuild_index()

    def get_active_turns(self) -> list[dict[str, Any]]:
        """Return turns currently in the active context window.

        Returns:
            List of dicts with ``id``, ``role``, ``content``, ``tokens``,
            ``importance`` keys.
        """
        return [
            {
                "id": t.id,
                "role": t.role,
                "content": t.content,
                "tokens": t.tokens,
                "importance": t.importance,
            }
            for t in self._active_turns
            if not t.evicted
        ]

    def get_system_prompt_addon(self) -> str:
        """Return the memory index text for injection into the system prompt.

        Returns:
            Formatted index string, or an empty string if no memories have
            been evicted yet.
        """
        return self._memory_index

    def build_memory_index(self) -> str:
        """Force a rebuild of the memory index and return it.

        Returns:
            Updated index string.
        """
        self._rebuild_index()
        return self._memory_index

    def reset_session(self) -> dict[str, Any]:
        """Clear the session layer in LLMFS and reset active turns.

        Returns:
            ``{"deleted": N, "session_id": "...", "status": "ok"}``.
        """
        result = self._mem.forget(layer="session")
        self._active_turns = []
        self._total_tokens = 0
        self._turn_counter = 0
        self._memory_index = ""
        self._evicted_count = 0
        self._eviction_rounds = 0
        logger.info("reset_session: session=%s deleted=%d", self._session_id, result["deleted"])
        return {**result, "session_id": self._session_id}

    def get_stats(self) -> dict[str, Any]:
        """Return context usage statistics.

        Returns:
            Dict with ``total_tokens``, ``max_tokens``, ``active_turns``,
            ``evicted_turns``, ``eviction_rounds``, ``session_id`` keys.
        """
        return {
            "session_id": self._session_id,
            "total_tokens": self._total_tokens,
            "max_tokens": self._max_tokens,
            "active_turns": len([t for t in self._active_turns if not t.evicted]),
            "evicted_turns": self._evicted_count,
            "eviction_rounds": self._eviction_rounds,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Evict lowest-importance turns until ``_total_tokens <= _target_tokens``."""
        self._eviction_rounds += 1

        # Re-score all active turns with updated total count (recency context)
        for i, t in enumerate(self._active_turns):
            if not t.evicted:
                t.importance = self._scorer.score(
                    t.content,
                    role=t.role,
                    turn_index=i,
                    total_turns=len(self._active_turns),
                )

        # Sort active turns by importance (ascending = evict cheapest first)
        eligible = sorted(
            [t for t in self._active_turns if not t.evicted],
            key=lambda t: t.importance,
        )

        for turn in eligible:
            if self._total_tokens <= self._target_tokens:
                break

            # Extract structured artifacts + archive full turn
            artifact_paths = self._extractor.extract(
                turn={"id": turn.id, "role": turn.role, "content": turn.content},
                mem=self._mem,
                session_id=self._session_id,
            )
            full_path = self._extractor.extract_and_store_full_turn(
                turn={"id": turn.id, "role": turn.role, "content": turn.content},
                mem=self._mem,
                session_id=self._session_id,
            )
            turn.artifact_paths = artifact_paths + [full_path]
            turn.evicted = True
            self._total_tokens -= turn.tokens
            self._evicted_count += 1

            logger.debug(
                "_evict: evicted turn=%s importance=%.2f tokens_freed=%d remaining=%d",
                turn.id, turn.importance, turn.tokens, self._total_tokens,
            )

    def _rebuild_index(self) -> None:
        """Rebuild the in-memory index from LLMFS."""
        try:
            self._memory_index = self._index_builder.build(
                session_id=self._session_id,
                mem=self._mem,
            )
        except Exception:
            logger.exception("Failed to rebuild memory index for session %s", self._session_id)

    def __repr__(self) -> str:
        return (
            f"ContextManager(session={self._session_id!r}, "
            f"tokens={self._total_tokens}/{self._max_tokens}, "
            f"active_turns={len(self._active_turns)})"
        )
