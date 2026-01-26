"""
Abstract base class for MemLearn metadata databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memlearn.types import MemFSLog, NodeMetadata, VersionSnapshot


class BaseDatabase(ABC):
    """Abstract base class for metadata storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database schema."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    # Node metadata operations
    @abstractmethod
    def save_metadata(self, metadata: NodeMetadata) -> None:
        """Save or update node metadata."""
        pass

    @abstractmethod
    def get_metadata(self, path: str) -> NodeMetadata | None:
        """Get metadata for a specific path."""
        pass

    @abstractmethod
    def delete_metadata(self, path: str) -> bool:
        """Delete metadata for a specific path. Returns True if deleted."""
        pass

    @abstractmethod
    def list_metadata(self, parent_path: str) -> list[NodeMetadata]:
        """List all metadata entries under a parent path."""
        pass

    @abstractmethod
    def move_metadata(self, old_path: str, new_path: str) -> bool:
        """Move metadata from old path to new path. Returns True if successful."""
        pass

    @abstractmethod
    def search_by_tags(self, tags: list[str]) -> list[NodeMetadata]:
        """Search metadata by tags."""
        pass

    # Log operations
    @abstractmethod
    def save_log(self, log: MemFSLog) -> None:
        """Save a log entry."""
        pass

    @abstractmethod
    def get_logs(
        self, limit: int = 100, offset: int = 0, path_filter: str | None = None
    ) -> list[MemFSLog]:
        """Get log entries with optional path filter."""
        pass

    # Version control operations
    @abstractmethod
    def save_snapshot(self, snapshot: VersionSnapshot) -> None:
        """Save a version snapshot."""
        pass

    @abstractmethod
    def get_snapshot(self, version_id: str) -> VersionSnapshot | None:
        """Get a specific version snapshot."""
        pass

    @abstractmethod
    def list_snapshots(self, limit: int = 50) -> list[VersionSnapshot]:
        """List recent version snapshots."""
        pass
