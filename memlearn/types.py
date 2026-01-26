"""
Core types for MemLearn - the filesystem-like memory architecture for LLM agents.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# =============================================================================
# Session Status
# =============================================================================


class SessionStatus(str, Enum):
    """Status of an agent session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


class MountSourceType(str, Enum):
    """Type of mount source."""

    AGENT = "agent"
    USER = "user"
    ORGANIZATION = "organization"
    EXTERNAL = "external"


# =============================================================================
# Agent and Session Entities
# =============================================================================


@dataclass
class Agent:
    """An agent entity with a unique name and ID."""

    agent_id: str
    name: str  # Unique, human-readable (e.g., "code-editor")
    created_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, extra: dict[str, Any] | None = None) -> "Agent":
        """Create a new agent with a generated UUID."""
        return cls(
            agent_id=str(uuid.uuid4()),
            name=name,
            created_at=time.time(),
            extra=extra or {},
        )


@dataclass
class Session:
    """A session representing a single agent run."""

    session_id: str
    agent_id: str  # Foreign key to Agent
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, agent_id: str, extra: dict[str, Any] | None = None) -> "Session":
        """Create a new session with a generated UUID."""
        return cls(
            session_id=str(uuid.uuid4()),
            agent_id=agent_id,
            started_at=time.time(),
            status=SessionStatus.ACTIVE,
            extra=extra or {},
        )

    def end(self, status: SessionStatus = SessionStatus.COMPLETED) -> None:
        """Mark the session as ended."""
        self.ended_at = time.time()
        self.status = status


@dataclass
class MountInfo:
    """Information about a mounted folder in MemFS."""

    mount_id: str
    agent_id: str  # The agent that has this mount
    mount_path: str  # Path in MemFS (e.g., "/mnt/users/user-123")
    source_type: MountSourceType
    source_ref: str  # ID of the source (e.g., another agent's ID)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        agent_id: str,
        mount_path: str,
        source_type: MountSourceType,
        source_ref: str,
    ) -> "MountInfo":
        """Create a new mount info with a generated UUID."""
        return cls(
            mount_id=str(uuid.uuid4()),
            agent_id=agent_id,
            mount_path=mount_path,
            source_type=source_type,
            source_ref=source_ref,
            created_at=time.time(),
        )


@dataclass
class CompactionMarker:
    """Marker indicating where conversation compaction occurred."""

    at_index: int  # Message index where compaction happened
    timestamp: float
    summary: str  # The summary that replaced earlier messages


@dataclass
class ConversationHistory:
    """Complete conversation history for a session."""

    session_id: str
    agent_id: str
    agent_name: str
    started_at: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    compaction_markers: list[CompactionMarker] = field(default_factory=list)

    @classmethod
    def create(
        cls, session_id: str, agent_id: str, agent_name: str
    ) -> "ConversationHistory":
        """Create a new conversation history."""
        return cls(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            started_at=time.time(),
            messages=[],
            compaction_markers=[],
        )

    def append_message(self, message: dict[str, Any]) -> None:
        """Append a message to the history."""
        self.messages.append(message)

    def add_compaction_marker(self, summary: str) -> None:
        """Add a compaction marker at the current position."""
        marker = CompactionMarker(
            at_index=len(self.messages),
            timestamp=time.time(),
            summary=summary,
        )
        self.compaction_markers.append(marker)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "messages": self.messages,
            "compaction_markers": [
                {
                    "at_index": m.at_index,
                    "timestamp": m.timestamp,
                    "summary": m.summary,
                }
                for m in self.compaction_markers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationHistory":
        """Create from dictionary (JSON deserialization)."""
        history = cls(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            started_at=data["started_at"],
            messages=data.get("messages", []),
        )
        for m in data.get("compaction_markers", []):
            history.compaction_markers.append(
                CompactionMarker(
                    at_index=m["at_index"],
                    timestamp=m["timestamp"],
                    summary=m["summary"],
                )
            )
        return history


class NodeType(str, Enum):
    """Type of node in MemFS."""

    FILE = "file"
    FOLDER = "folder"


class FileType(str, Enum):
    """Type of file content."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE = "code"
    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


@dataclass
class Permissions:
    """Permission settings for a MemFS node."""

    readable: bool = True
    writable: bool = True
    executable: bool = False


@dataclass
class Timestamps:
    """Timestamps for a MemFS node."""

    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)

    def touch_access(self) -> None:
        """Update access time."""
        self.accessed_at = time.time()

    def touch_modify(self) -> None:
        """Update modification time (and access time)."""
        now = time.time()
        self.accessed_at = now
        self.modified_at = now


@dataclass
class Chunk:
    """A chunk of file content with its embedding."""

    content: str
    start_line: int
    end_line: int
    content_hash: str
    embedding: list[float] | None = None

    @classmethod
    def from_content(cls, content: str, start_line: int, end_line: int) -> "Chunk":
        """Create a chunk from content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            content=content,
            start_line=start_line,
            end_line=end_line,
            content_hash=content_hash,
        )


@dataclass
class NodeMetadata:
    """Metadata for a MemFS node (file or folder)."""

    path: str
    node_type: NodeType
    timestamps: Timestamps = field(default_factory=Timestamps)
    permissions: Permissions = field(default_factory=Permissions)
    owner: str = "system"
    file_type: FileType = FileType.TEXT
    description: str = ""
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    chunks: list[Chunk] = field(default_factory=list)
    size_bytes: int = 0
    line_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_file(self) -> bool:
        return self.node_type == NodeType.FILE

    @property
    def is_folder(self) -> bool:
        return self.node_type == NodeType.FOLDER


@dataclass
class MemFSLog:
    """Log entry for MemFS operations."""

    timestamp: float
    operation: str
    path: str
    agent_id: str
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None

    @classmethod
    def create(
        cls,
        operation: str,
        path: str,
        agent_id: str = "system",
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> "MemFSLog":
        return cls(
            timestamp=time.time(),
            operation=operation,
            path=path,
            agent_id=agent_id,
            details=details or {},
            success=success,
            error_message=error_message,
        )


@dataclass
class SearchResult:
    """Result from a MemFS search operation."""

    path: str
    score: float
    content_snippet: str | None = None
    chunk_index: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: NodeMetadata | None = None


@dataclass
class VersionSnapshot:
    """A snapshot of MemFS state for version control."""

    version_id: str
    timestamp: float
    description: str
    changed_paths: list[str]
    agent_id: str = "system"


# Tool result types for agent responses
ToolResultStatus = Literal["success", "error"]


@dataclass
class ToolResult:
    """Standard result format for MemFS tool operations."""

    status: ToolResultStatus
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result
