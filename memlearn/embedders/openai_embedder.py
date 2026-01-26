"""
OpenAI embedding provider for MemLearn.
"""

from __future__ import annotations

from openai import OpenAI

from memlearn.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-based embedding provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: The embedding model to use.
            dimensions: The embedding dimensions (for models that support it).
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Handle empty text
        if not text.strip():
            return [0.0] * self._dimensions

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self._dimensions,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        if not texts:
            return []

        # Handle empty texts by replacing with placeholder
        processed_texts = [t if t.strip() else " " for t in texts]

        response = self.client.embeddings.create(
            model=self.model,
            input=processed_texts,
            dimensions=self._dimensions,
        )

        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]
