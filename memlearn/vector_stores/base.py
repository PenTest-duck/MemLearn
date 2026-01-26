"""
Abstract base class for MemLearn vector stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""

    id: str
    score: float
    metadata: dict


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the vector store connection."""
        pass

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Add vectors to the store.

        Args:
            ids: Unique identifiers for each vector.
            embeddings: The embedding vectors.
            metadatas: Optional metadata for each vector.
            documents: Optional document text for each vector.
        """
        pass

    @abstractmethod
    def update(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Update vectors in the store.

        Args:
            ids: Unique identifiers for vectors to update.
            embeddings: New embedding vectors (optional).
            metadatas: New metadata (optional).
            documents: New document text (optional).
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete vectors from the store.

        Args:
            ids: Unique identifiers for vectors to delete.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of search results sorted by similarity.
        """
        pass

    @abstractmethod
    def get(self, ids: list[str]) -> list[VectorSearchResult]:
        """
        Get vectors by ID.

        Args:
            ids: Unique identifiers for vectors to retrieve.

        Returns:
            List of results.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pass
