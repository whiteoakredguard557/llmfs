"""
Abstract base class for LLMFS embedders.

All embedders (local, OpenAI, etc.) implement this interface so the rest of
the system is embedder-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["EmbedderBase"]


class EmbedderBase(ABC):
    """Abstract interface for text embedders.

    Subclasses must implement :meth:`embed` and :meth:`embed_batch`.
    The :attr:`model_name` property is used for logging and diagnostics.

    Example::

        class MyEmbedder(EmbedderBase):
            @property
            def model_name(self) -> str:
                return "my-model"

            def embed(self, text: str) -> list[float]:
                ...

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed(t) for t in texts]
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the underlying model."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: The text to embed. Will be truncated by the model if it
                exceeds the model's context length.

        Returns:
            Float vector of length :attr:`embedding_dim`.

        Raises:
            EmbedderError: If the model fails.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input string.

        Raises:
            EmbedderError: If the model fails.
        """
