"""
Core types for MemLearn - the filesystem-like memory architecture for LLM agents.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


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
    def from_content(
        cls, content: str, start_line: int, end_line: int
    ) -> "Chunk":
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
