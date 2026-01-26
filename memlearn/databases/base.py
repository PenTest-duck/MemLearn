"""
Abstract base class for MemLearn metadata databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memlearn.types import (
    Agent,
    MemFSLog,
    MountInfo,
    NodeMetadata,
    Session,
    SessionStatus,
    VersionSnapshot,
)


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

    # =========================================================================
    # Agent operations
    # =========================================================================

    @abstractmethod
    def create_agent(self, agent: Agent) -> None:
        """Create a new agent."""
        pass

    @abstractmethod
    def get_agent_by_id(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        pass

    @abstractmethod
    def get_agent_by_name(self, name: str) -> Agent | None:
        """Get an agent by name."""
        pass

    @abstractmethod
    def list_agents(self) -> list[Agent]:
        """List all agents."""
        pass

    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent. Returns True if deleted."""
        pass

    # =========================================================================
    # Session operations
    # =========================================================================

    @abstractmethod
    def create_session(self, session: Session) -> None:
        """Create a new session."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        pass

    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update an existing session."""
        pass

    @abstractmethod
    def end_session(
        self, session_id: str, status: SessionStatus = SessionStatus.COMPLETED
    ) -> None:
        """End a session with the given status."""
        pass

    @abstractmethod
    def get_sessions_for_agent(
        self, agent_id: str, limit: int = 50
    ) -> list[Session]:
        """Get sessions for an agent, most recent first."""
        pass

    @abstractmethod
    def get_active_session_for_agent(self, agent_id: str) -> Session | None:
        """Get the currently active session for an agent, if any."""
        pass

    # =========================================================================
    # Mount operations
    # =========================================================================

    @abstractmethod
    def create_mount(self, mount: MountInfo) -> None:
        """Create a new mount record."""
        pass

    @abstractmethod
    def get_mounts_for_agent(self, agent_id: str) -> list[MountInfo]:
        """Get all mounts for an agent."""
        pass

    @abstractmethod
    def delete_mount(self, mount_id: str) -> bool:
        """Delete a mount record. Returns True if deleted."""
        pass

    @abstractmethod
    def delete_mounts_for_agent(self, agent_id: str) -> int:
        """Delete all mounts for an agent. Returns count deleted."""
        pass

    # =========================================================================
    # Node metadata operations
    # =========================================================================

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

    # =========================================================================
    # Log operations
    # =========================================================================

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

    # =========================================================================
    # Version control operations
    # =========================================================================

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
