"""
Core types for MemLearn - the filesystem-like memory architecture for LLM agents.
"""

from __future__ import annotations

import datetime
import hashlib
import re
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
    # Dynamic memory note - a concise summary of what's stored in memory
    # This is auto-updated on spindown and can be injected into system prompts
    memory_note: str = "This memory is empty. No files or insights have been stored yet."

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

    def to_markdown(self) -> str:
        """Convert conversation history to agent-friendly markdown format.

        The markdown format includes:
        - YAML frontmatter with session metadata
        - Each message as a level-2 heading with timestamp and role
        - Tool calls and results formatted with tool name
        - Compaction markers as horizontal rules with summaries

        Returns:
            Markdown string representation of the conversation history.
        """
        lines: list[str] = []

        # YAML frontmatter
        started_dt = datetime.datetime.fromtimestamp(self.started_at)
        lines.append("---")
        lines.append(f"session_id: {self.session_id}")
        lines.append(f"agent_id: {self.agent_id}")
        lines.append(f"agent_name: {self.agent_name}")
        lines.append(f"started_at: {started_dt.isoformat()}")
        lines.append("---")
        lines.append("")
        lines.append("# Conversation History")
        lines.append("")

        # Track compaction markers by index
        compaction_by_index = {m.at_index: m for m in self.compaction_markers}

        for i, msg in enumerate(self.messages):
            # Check for compaction marker before this message
            if i in compaction_by_index:
                marker = compaction_by_index[i]
                marker_dt = datetime.datetime.fromtimestamp(marker.timestamp)
                lines.append("---")
                lines.append("")
                lines.append(
                    f"### Compaction Marker [{marker_dt.strftime('%Y-%m-%d %H:%M:%S')}]"
                )
                lines.append("")
                lines.append(marker.summary)
                lines.append("")
                lines.append("---")
                lines.append("")

            # Format message
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp")

            # Format timestamp
            if timestamp:
                msg_dt = datetime.datetime.fromtimestamp(timestamp)
                time_str = msg_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = "unknown"

            # Handle different message types
            if role == "tool_call":
                tool_name = msg.get("tool_name", "unknown_tool")
                lines.append(f"## [{time_str}] tool_call")
                lines.append("")
                lines.append(f"**Tool:** `{tool_name}`")
                lines.append("")
                if content:
                    lines.append("```json")
                    lines.append(content)
                    lines.append("```")
            elif role == "tool_result":
                tool_name = msg.get("tool_name", "unknown_tool")
                lines.append(f"## [{time_str}] tool_result")
                lines.append("")
                lines.append(f"**Tool:** `{tool_name}`")
                lines.append("")
                lines.append(content)
            else:
                lines.append(f"## [{time_str}] {role}")
                lines.append("")
                lines.append(content)

            lines.append("")

        # Handle compaction marker at the end (after all messages)
        if len(self.messages) in compaction_by_index:
            marker = compaction_by_index[len(self.messages)]
            marker_dt = datetime.datetime.fromtimestamp(marker.timestamp)
            lines.append("---")
            lines.append("")
            lines.append(
                f"### Compaction Marker [{marker_dt.strftime('%Y-%m-%d %H:%M:%S')}]"
            )
            lines.append("")
            lines.append(marker.summary)
            lines.append("")
            lines.append("---")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, markdown: str) -> "ConversationHistory":
        """Parse markdown format back to ConversationHistory.

        Args:
            markdown: The markdown string to parse.

        Returns:
            ConversationHistory object.

        Raises:
            ValueError: If the markdown format is invalid.
        """
        # Parse YAML frontmatter
        frontmatter_match = re.match(
            r"^---\n(.*?)\n---\n", markdown, re.DOTALL
        )
        if not frontmatter_match:
            raise ValueError("Invalid markdown: missing YAML frontmatter")

        frontmatter = frontmatter_match.group(1)
        body = markdown[frontmatter_match.end() :]

        # Parse frontmatter fields
        session_id = ""
        agent_id = ""
        agent_name = ""
        started_at = 0.0

        for line in frontmatter.split("\n"):
            if line.startswith("session_id:"):
                session_id = line.split(":", 1)[1].strip()
            elif line.startswith("agent_id:"):
                agent_id = line.split(":", 1)[1].strip()
            elif line.startswith("agent_name:"):
                agent_name = line.split(":", 1)[1].strip()
            elif line.startswith("started_at:"):
                dt_str = line.split(":", 1)[1].strip()
                try:
                    dt = datetime.datetime.fromisoformat(dt_str)
                    started_at = dt.timestamp()
                except ValueError:
                    started_at = time.time()

        history = cls(
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            started_at=started_at,
            messages=[],
            compaction_markers=[],
        )

        # Parse messages and compaction markers
        # Pattern for message headers: ## [timestamp] role
        message_pattern = re.compile(
            r"^## \[([^\]]+)\] (\w+)\s*$", re.MULTILINE
        )
        # Pattern for compaction markers: ### Compaction Marker [timestamp]
        compaction_pattern = re.compile(
            r"^### Compaction Marker \[([^\]]+)\]\s*$", re.MULTILINE
        )

        # Find all message and compaction marker positions
        elements: list[tuple[int, str, Any]] = []  # (pos, type, match)

        for match in message_pattern.finditer(body):
            elements.append((match.start(), "message", match))

        for match in compaction_pattern.finditer(body):
            elements.append((match.start(), "compaction", match))

        # Sort by position
        elements.sort(key=lambda x: x[0])

        # Process each element
        for i, (pos, elem_type, match) in enumerate(elements):
            # Find content end (start of next element or end of body)
            if i + 1 < len(elements):
                content_end = elements[i + 1][0]
            else:
                content_end = len(body)

            # Extract content after the header
            header_end = match.end()
            content = body[header_end:content_end].strip()

            if elem_type == "message":
                time_str = match.group(1)
                role = match.group(2)

                # Parse timestamp
                timestamp = None
                if time_str != "unknown":
                    try:
                        dt = datetime.datetime.strptime(
                            time_str, "%Y-%m-%d %H:%M:%S"
                        )
                        timestamp = dt.timestamp()
                    except ValueError:
                        pass

                # Parse tool name if present
                tool_name = None
                msg_content = content

                if role in ("tool_call", "tool_result"):
                    # Look for **Tool:** `tool_name`
                    tool_match = re.match(
                        r"\*\*Tool:\*\* `([^`]+)`\s*\n?", content
                    )
                    if tool_match:
                        tool_name = tool_match.group(1)
                        msg_content = content[tool_match.end() :].strip()

                        # For tool_call, extract JSON from code block
                        if role == "tool_call":
                            json_match = re.match(
                                r"```json\n(.*?)\n```",
                                msg_content,
                                re.DOTALL,
                            )
                            if json_match:
                                msg_content = json_match.group(1)

                # Build message dict
                msg: dict[str, Any] = {
                    "role": role,
                    "content": msg_content,
                }
                if timestamp is not None:
                    msg["timestamp"] = timestamp
                if tool_name:
                    msg["tool_name"] = tool_name

                history.messages.append(msg)

            elif elem_type == "compaction":
                time_str = match.group(1)

                # Parse timestamp
                try:
                    dt = datetime.datetime.strptime(
                        time_str, "%Y-%m-%d %H:%M:%S"
                    )
                    timestamp = dt.timestamp()
                except ValueError:
                    timestamp = time.time()

                # Remove surrounding --- if present
                summary = content
                if summary.startswith("---"):
                    summary = summary[3:].strip()
                if summary.endswith("---"):
                    summary = summary[:-3].strip()

                marker = CompactionMarker(
                    at_index=len(history.messages),
                    timestamp=timestamp,
                    summary=summary,
                )
                history.compaction_markers.append(marker)

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
