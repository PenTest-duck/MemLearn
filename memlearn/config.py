"""
Configuration management for MemLearn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# Provider type definitions
DatabaseProvider = Literal["sqlite", "postgres"]
VectorStoreProvider = Literal["chroma", "qdrant"]
EmbedderProvider = Literal["openai", "voyage", "gemini"]
RerankerProvider = Literal["cohere", "none"]
SandboxProvider = Literal["local", "e2b"]
AgentProvider = Literal["openai", "anthropic", "gemini"]


@dataclass
class DatabaseConfig:
    """Configuration for the metadata database."""

    provider: DatabaseProvider = "sqlite"
    # SQLite
    sqlite_path: str | None = None  # None means in-memory
    # Postgres
    postgres_url: str | None = None


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""

    provider: VectorStoreProvider = "chroma"
    # Chroma
    chroma_path: str | None = None  # None means in-memory
    chroma_collection_name: str = "memfs_embeddings"
    # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "memfs_embeddings"


@dataclass
class EmbedderConfig:
    """Configuration for the embedding model."""

    provider: EmbedderProvider = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_key: str | None = None
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class RerankerConfig:
    """Configuration for the reranker."""

    provider: RerankerProvider = "cohere"
    model: str = "rerank-v3.5"
    api_key: str | None = None
    top_n: int = 5


@dataclass
class SandboxConfig:
    """Configuration for the ephemeral filesystem sandbox."""

    provider: SandboxProvider = "local"
    # Local
    temp_dir_prefix: str = "memfs-"
    # E2B
    e2b_api_key: str | None = None
    e2b_template: str = "base"


@dataclass
class AgentConfig:
    """Configuration for the target agent framework."""

    provider: AgentProvider = "openai"
    # Tool configuration
    tool_prefix: str = "memfs"
    include_line_numbers: bool = True
    max_read_lines: int = 200
    truncate_long_files: bool = True
    edit_method: Literal["str_replace", "udiff"] = "str_replace"


@dataclass
class MemLearnConfig:
    """Main configuration for MemLearn."""

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Session settings
    agent_id: str = "default"
    session_id: str | None = None
    auto_embed: bool = True
    auto_log: bool = True

    @classmethod
    def from_env(cls) -> "MemLearnConfig":
        """Create configuration from environment variables."""
        config = cls()

        # API keys from environment
        config.embedder.api_key = os.getenv("OPENAI_API_KEY")
        config.reranker.api_key = os.getenv("COHERE_API_KEY")
        config.sandbox.e2b_api_key = os.getenv("E2B_API_KEY")

        return config

    @classmethod
    def default_local(cls) -> "MemLearnConfig":
        """Create a default local configuration for development."""
        return cls(
            database=DatabaseConfig(provider="sqlite", sqlite_path=None),
            vector_store=VectorStoreConfig(provider="chroma", chroma_path=None),
            embedder=EmbedderConfig(
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            reranker=RerankerConfig(
                provider="cohere",
                api_key=os.getenv("COHERE_API_KEY"),
            ),
            sandbox=SandboxConfig(provider="local"),
            agent=AgentConfig(provider="openai"),
        )
