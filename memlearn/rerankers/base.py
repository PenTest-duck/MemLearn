"""
Abstract base class for MemLearn rerankers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Result from a reranking operation."""

    index: int
    score: float
    document: str


class BaseReranker(ABC):
    """Abstract base class for reranking providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_n: Number of top results to return. None returns all.

        Returns:
            List of rerank results sorted by relevance score.
        """
        pass


class NoOpReranker(BaseReranker):
    """A no-op reranker that returns documents in original order."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return documents in original order with uniform scores."""
        results = [
            RerankResult(index=i, score=1.0 - (i * 0.01), document=doc)
            for i, doc in enumerate(documents)
        ]

        if top_n is not None:
            results = results[:top_n]

        return results
