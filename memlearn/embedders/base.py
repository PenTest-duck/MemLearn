"""
Abstract base class for MemLearn embedding providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """
        pass

    def chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> list[tuple[str, int, int]]:
        """
        Split text into chunks for embedding.

        Args:
            text: The text to chunk.
            chunk_size: Approximate number of characters per chunk.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of tuples (chunk_text, start_line, end_line).
        """
        lines = text.split("\n")
        chunks = []
        current_chunk_lines: list[str] = []
        current_chunk_size = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline
            current_chunk_lines.append(line)
            current_chunk_size += line_size

            if current_chunk_size >= chunk_size:
                # Save current chunk
                chunk_text = "\n".join(current_chunk_lines)
                chunks.append((chunk_text, start_line, i))

                # Start new chunk with overlap
                overlap_lines = []
                overlap_size = 0
                for j in range(len(current_chunk_lines) - 1, -1, -1):
                    overlap_size += len(current_chunk_lines[j]) + 1
                    overlap_lines.insert(0, current_chunk_lines[j])
                    if overlap_size >= overlap:
                        break

                current_chunk_lines = overlap_lines
                current_chunk_size = overlap_size
                start_line = i - len(overlap_lines) + 1

        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append((chunk_text, start_line, len(lines)))

        return chunks
