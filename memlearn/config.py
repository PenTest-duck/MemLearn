"""
Configuration management for MemLearn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Provider type definitions
DatabaseProvider = Literal["sqlite", "postgres"]
VectorStoreProvider = Literal["chroma", "qdrant"]
EmbedderProvider = Literal["openai", "voyage", "gemini"]
RerankerProvider = Literal["cohere", "none"]
SandboxProvider = Literal["local", "e2b"]
AgentProvider = Literal["openai", "anthropic", "gemini"]
LLMProvider = Literal["openai", "anthropic", "gemini"]

# Default paths
DEFAULT_MEMLEARN_HOME = Path.home() / ".memlearn"


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
    # Persistent storage path for syncing between sessions
    # Default: ~/.memlearn/persistent
    persistent_storage_path: str | None = None
    # E2B
    e2b_api_key: str | None = None
    e2b_template: str = "base"
    # Version control - enables git-based versioning for undo support
    version_control_enabled: bool = True

    def get_persistent_storage_path(self) -> Path:
        """Get the persistent storage path, using default if not set."""
        if self.persistent_storage_path:
            return Path(self.persistent_storage_path)
        return DEFAULT_MEMLEARN_HOME / "persistent"

    def get_agent_persistent_path(self, agent_id: str) -> Path:
        """Get the persistent storage path for a specific agent."""
        return self.get_persistent_storage_path() / "agents" / agent_id


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
class LLMConfig:
    """Configuration for the LLM provider used for internal tasks.

    This configures the LLM used for MemFS internal operations like:
    - Conversation summarization on spindown
    - Memory note generation for system prompt injection
    - Memory reflection and consolidation (future)
    - Automatic tagging and categorization (future)
    """

    provider: LLMProvider = "openai"
    # Model for summarization and reflection tasks
    model: str = "gpt-5-mini"
    api_key: str | None = None
    # Summarization settings
    summarize_on_spindown: bool = True
    max_summary_tokens: int = 16000  # Reasoning models need more tokens
    # Memory note settings
    # The memory note is a concise summary of what's stored in memory,
    # auto-updated on spindown for injection into agent system prompts
    update_memory_note_on_spindown: bool = True
    memory_note_max_tokens: int = 8000  # Reasoning models need more tokens
    # Temperature for different task types
    summarization_temperature: float = 0.3
    reflection_temperature: float = 0.5


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
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Session settings (set during spinup)
    agent_id: str | None = None
    session_id: str | None = None
    agent_name: str | None = None
    auto_embed: bool = True
    auto_log: bool = True

    # Debug mode - enables verbose logging for diagnosing issues
    debug: bool = False

    def get_memlearn_home(self) -> Path:
        """Get the MemLearn home directory."""
        return DEFAULT_MEMLEARN_HOME

    @classmethod
    def from_env(cls) -> "MemLearnConfig":
        """Create configuration from environment variables."""
        config = cls()

        # API keys from environment
        config.embedder.api_key = os.getenv("OPENAI_API_KEY")
        config.reranker.api_key = os.getenv("COHERE_API_KEY")
        config.sandbox.e2b_api_key = os.getenv("E2B_API_KEY")
        config.llm.api_key = os.getenv("OPENAI_API_KEY")

        return config

    @classmethod
    def default_local(cls) -> "MemLearnConfig":
        """Create a default local configuration for development (in-memory, no persistence)."""
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
            llm=LLMConfig(
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        )

    @classmethod
    def default_persistent(cls) -> "MemLearnConfig":
        """Create a default configuration with persistence enabled.

        Uses ~/.memlearn/ for all persistent storage:
        - ~/.memlearn/memlearn.db (SQLite database)
        - ~/.memlearn/chroma/ (Vector store)
        - ~/.memlearn/persistent/agents/{agent_id}/ (Agent filesystems)
        """
        home = DEFAULT_MEMLEARN_HOME
        return cls(
            database=DatabaseConfig(
                provider="sqlite",
                sqlite_path=str(home / "memlearn.db"),
            ),
            vector_store=VectorStoreConfig(
                provider="chroma",
                chroma_path=str(home / "chroma"),
            ),
            embedder=EmbedderConfig(
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            reranker=RerankerConfig(
                provider="cohere",
                api_key=os.getenv("COHERE_API_KEY"),
            ),
            sandbox=SandboxConfig(
                provider="local",
                persistent_storage_path=str(home / "persistent"),
            ),
            agent=AgentConfig(provider="openai"),
            llm=LLMConfig(
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        )
