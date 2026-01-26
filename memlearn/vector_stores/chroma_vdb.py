"""
ChromaDB vector store implementation for MemLearn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings

from memlearn.vector_stores.base import BaseVectorStore, VectorSearchResult

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store for MemLearn."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "memfs_embeddings",
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage. None for in-memory.
            collection_name: Name of the collection to use.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    def initialize(self) -> None:
        """Initialize the vector store."""
        if self.persist_directory:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # Use EphemeralClient for in-memory storage (not chromadb.Client)
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Close the vector store connection."""
        # ChromaDB doesn't require explicit close for in-memory
        self._collection = None
        self._client = None

    def _clean_metadata(self, metadatas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean metadata to only include ChromaDB-compatible values."""
        cleaned = []
        for meta in metadatas:
            # ChromaDB only supports str, int, float, bool in metadata
            cleaned_meta = {
                k: v
                for k, v in meta.items()
                if v is not None and isinstance(v, (str, int, float, bool))
            }
            cleaned.append(cleaned_meta)
        return cleaned

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Add vectors to the store."""
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        kwargs: dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings,
        }

        if metadatas is not None:
            kwargs["metadatas"] = self._clean_metadata(metadatas)

        if documents is not None:
            kwargs["documents"] = documents

        self._collection.add(**kwargs)

    def update(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Update vectors in the store."""
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        kwargs: dict[str, Any] = {"ids": ids}

        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        if metadatas is not None:
            kwargs["metadatas"] = self._clean_metadata(metadatas)

        if documents is not None:
            kwargs["documents"] = documents

        self._collection.update(**kwargs)

    def delete(self, ids: list[str]) -> None:
        """Delete vectors from the store."""
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        self._collection.delete(ids=ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self._collection.query(**kwargs)

        search_results = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = (
                results["distances"][0] if results["distances"] else [0.0] * len(ids)
            )
            metadatas = (
                results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            )

            for i, id_ in enumerate(ids):
                # ChromaDB returns distance, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                score = 1.0 - distances[i] if distances else 1.0
                search_results.append(
                    VectorSearchResult(
                        id=id_,
                        score=score,
                        metadata=metadatas[i] if metadatas else {},
                    )
                )

        return search_results

    def get(self, ids: list[str]) -> list[VectorSearchResult]:
        """Get vectors by ID."""
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        results = self._collection.get(ids=ids, include=["embeddings", "metadatas"])

        search_results = []
        if results["ids"]:
            for i, id_ in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                search_results.append(
                    VectorSearchResult(
                        id=id_,
                        score=1.0,  # Exact match
                        metadata=metadata,
                    )
                )

        return search_results

    def count(self) -> int:
        """Return the number of vectors in the store."""
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        return self._collection.count()

    def clear(self) -> None:
        """Clear all vectors from the store."""
        if self._client is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
