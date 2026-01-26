"""
Cohere reranker implementation for MemLearn.
"""

from __future__ import annotations

import cohere

from memlearn.rerankers.base import BaseReranker, RerankResult


class CohereReranker(BaseReranker):
    """Cohere-based reranking provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-v3.5",
    ):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key. If None, uses CO_API_KEY env var.
            model: The reranking model to use.
        """
        self.client = cohere.Client(api_key=api_key)
        self.model = model

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
        if not documents:
            return []

        if top_n is None:
            top_n = len(documents)

        response = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=self.model,
        )

        return [
            RerankResult(
                index=result.index,
                score=result.relevance_score,
                document=documents[result.index],
            )
            for result in response.results
        ]
