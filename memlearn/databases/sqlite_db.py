"""
SQLite implementation for MemLearn metadata storage.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from memlearn.databases.base import BaseDatabase
from memlearn.types import (
    Agent,
    Chunk,
    FileType,
    MemFSLog,
    MountInfo,
    MountSourceType,
    NodeMetadata,
    NodeType,
    Permissions,
    Session,
    SessionStatus,
    Timestamps,
    VersionSnapshot,
)


class SQLiteDatabase(BaseDatabase):
    """SQLite-based metadata storage for MemLearn."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize SQLite database.

        Args:
            db_path: Path to SQLite database file. None for in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self.conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Initialize the database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        assert self.conn is not None

        cursor = self.conn.cursor()

        # Agents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                extra TEXT NOT NULL DEFAULT '{}',
                memory_note TEXT NOT NULL DEFAULT 'This memory is empty. No files or insights have been stored yet.'
            )
        """)

        # Migration: Add memory_note column if it doesn't exist (for existing databases)
        cursor.execute("PRAGMA table_info(agents)")
        columns = [col[1] for col in cursor.fetchall()]
        if "memory_note" not in columns:
            cursor.execute("""
                ALTER TABLE agents ADD COLUMN memory_note TEXT 
                NOT NULL DEFAULT 'This memory is empty. No files or insights have been stored yet.'
            """)

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                status TEXT NOT NULL DEFAULT 'active',
                extra TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        """)

        # Mounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mounts (
                mount_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                mount_path TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        """)

        # Node metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS node_metadata (
                path TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                modified_at REAL NOT NULL,
                readable INTEGER NOT NULL DEFAULT 1,
                writable INTEGER NOT NULL DEFAULT 1,
                executable INTEGER NOT NULL DEFAULT 0,
                owner TEXT NOT NULL DEFAULT 'system',
                file_type TEXT NOT NULL DEFAULT 'text',
                description TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                embedding TEXT,
                chunks TEXT NOT NULL DEFAULT '[]',
                size_bytes INTEGER NOT NULL DEFAULT 0,
                line_count INTEGER NOT NULL DEFAULT 0,
                extra TEXT NOT NULL DEFAULT '{}'
            )
        """)

        # Logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                path TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                success INTEGER NOT NULL DEFAULT 1,
                error_message TEXT
            )
        """)

        # Version snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_snapshots (
                version_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                description TEXT NOT NULL,
                changed_paths TEXT NOT NULL DEFAULT '[]',
                agent_id TEXT NOT NULL DEFAULT 'system'
            )
        """)

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mounts_agent_id ON mounts(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_path ON logs(path)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON version_snapshots(timestamp DESC)"
        )

        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # =========================================================================
    # Agent operations
    # =========================================================================

    def create_agent(self, agent: Agent) -> None:
        """Create a new agent."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO agents (agent_id, name, created_at, extra, memory_note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                agent.agent_id,
                agent.name,
                agent.created_at,
                json.dumps(agent.extra),
                agent.memory_note,
            ),
        )
        self.conn.commit()

    def get_agent_by_id(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return Agent(
            agent_id=row["agent_id"],
            name=row["name"],
            created_at=row["created_at"],
            extra=json.loads(row["extra"]),
            memory_note=row["memory_note"],
        )

    def get_agent_by_name(self, name: str) -> Agent | None:
        """Get an agent by name."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents WHERE name = ?", (name,))
        row = cursor.fetchone()

        if row is None:
            return None

        return Agent(
            agent_id=row["agent_id"],
            name=row["name"],
            created_at=row["created_at"],
            extra=json.loads(row["extra"]),
            memory_note=row["memory_note"],
        )

    def list_agents(self) -> list[Agent]:
        """List all agents."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM agents ORDER BY created_at DESC")

        return [
            Agent(
                agent_id=row["agent_id"],
                name=row["name"],
                created_at=row["created_at"],
                extra=json.loads(row["extra"]),
                memory_note=row["memory_note"],
            )
            for row in cursor.fetchall()
        ]

    def update_agent(self, agent: Agent) -> None:
        """Update an existing agent."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE agents
            SET name = ?, extra = ?, memory_note = ?
            WHERE agent_id = ?
            """,
            (
                agent.name,
                json.dumps(agent.extra),
                agent.memory_note,
                agent.agent_id,
            ),
        )
        self.conn.commit()

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent. Returns True if deleted."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # Session operations
    # =========================================================================

    def create_session(self, session: Session) -> None:
        """Create a new session."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (session_id, agent_id, started_at, ended_at, status, extra)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.agent_id,
                session.started_at,
                session.ended_at,
                session.status.value,
                json.dumps(session.extra),
            ),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return Session(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=SessionStatus(row["status"]),
            extra=json.loads(row["extra"]),
        )

    def update_session(self, session: Session) -> None:
        """Update an existing session."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE sessions
            SET started_at = ?, ended_at = ?, status = ?, extra = ?
            WHERE session_id = ?
            """,
            (
                session.started_at,
                session.ended_at,
                session.status.value,
                json.dumps(session.extra),
                session.session_id,
            ),
        )
        self.conn.commit()

    def end_session(
        self, session_id: str, status: SessionStatus = SessionStatus.COMPLETED
    ) -> None:
        """End a session with the given status."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE sessions
            SET ended_at = ?, status = ?
            WHERE session_id = ?
            """,
            (time.time(), status.value, session_id),
        )
        self.conn.commit()

    def get_sessions_for_agent(
        self, agent_id: str, limit: int = 50
    ) -> list[Session]:
        """Get sessions for an agent, most recent first."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sessions
            WHERE agent_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (agent_id, limit),
        )

        return [
            Session(
                session_id=row["session_id"],
                agent_id=row["agent_id"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                status=SessionStatus(row["status"]),
                extra=json.loads(row["extra"]),
            )
            for row in cursor.fetchall()
        ]

    def get_active_session_for_agent(self, agent_id: str) -> Session | None:
        """Get the currently active session for an agent, if any."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sessions
            WHERE agent_id = ? AND status = 'active'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (agent_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return Session(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=SessionStatus(row["status"]),
            extra=json.loads(row["extra"]),
        )

    # =========================================================================
    # Mount operations
    # =========================================================================

    def create_mount(self, mount: MountInfo) -> None:
        """Create a new mount record."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO mounts (mount_id, agent_id, mount_path, source_type, source_ref, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                mount.mount_id,
                mount.agent_id,
                mount.mount_path,
                mount.source_type.value,
                mount.source_ref,
                mount.created_at,
            ),
        )
        self.conn.commit()

    def get_mounts_for_agent(self, agent_id: str) -> list[MountInfo]:
        """Get all mounts for an agent."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM mounts WHERE agent_id = ? ORDER BY created_at",
            (agent_id,),
        )

        return [
            MountInfo(
                mount_id=row["mount_id"],
                agent_id=row["agent_id"],
                mount_path=row["mount_path"],
                source_type=MountSourceType(row["source_type"]),
                source_ref=row["source_ref"],
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    def delete_mount(self, mount_id: str) -> bool:
        """Delete a mount record. Returns True if deleted."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM mounts WHERE mount_id = ?", (mount_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_mounts_for_agent(self, agent_id: str) -> int:
        """Delete all mounts for an agent. Returns count deleted."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM mounts WHERE agent_id = ?", (agent_id,))
        self.conn.commit()
        return cursor.rowcount

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _serialize_embedding(self, embedding: list[float] | None) -> str | None:
        """Serialize embedding to JSON string."""
        if embedding is None:
            return None
        return json.dumps(embedding)

    def _deserialize_embedding(self, data: str | None) -> list[float] | None:
        """Deserialize embedding from JSON string."""
        if data is None:
            return None
        return json.loads(data)

    def _serialize_chunks(self, chunks: list[Chunk]) -> str:
        """Serialize chunks to JSON string."""
        return json.dumps([
            {
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content_hash": c.content_hash,
                "embedding": c.embedding,
            }
            for c in chunks
        ])

    def _deserialize_chunks(self, data: str) -> list[Chunk]:
        """Deserialize chunks from JSON string."""
        items = json.loads(data)
        return [
            Chunk(
                content=item["content"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                content_hash=item["content_hash"],
                embedding=item.get("embedding"),
            )
            for item in items
        ]

    def _row_to_metadata(self, row: sqlite3.Row) -> NodeMetadata:
        """Convert a database row to NodeMetadata."""
        return NodeMetadata(
            path=row["path"],
            node_type=NodeType(row["node_type"]),
            timestamps=Timestamps(
                created_at=row["created_at"],
                accessed_at=row["accessed_at"],
                modified_at=row["modified_at"],
            ),
            permissions=Permissions(
                readable=bool(row["readable"]),
                writable=bool(row["writable"]),
                executable=bool(row["executable"]),
            ),
            owner=row["owner"],
            file_type=FileType(row["file_type"]),
            description=row["description"],
            tags=json.loads(row["tags"]),
            embedding=self._deserialize_embedding(row["embedding"]),
            chunks=self._deserialize_chunks(row["chunks"]),
            size_bytes=row["size_bytes"],
            line_count=row["line_count"],
            extra=json.loads(row["extra"]),
        )

    def save_metadata(self, metadata: NodeMetadata) -> None:
        """Save or update node metadata."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO node_metadata (
                path, node_type, created_at, accessed_at, modified_at,
                readable, writable, executable, owner, file_type,
                description, tags, embedding, chunks, size_bytes,
                line_count, extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.path,
                metadata.node_type.value,
                metadata.timestamps.created_at,
                metadata.timestamps.accessed_at,
                metadata.timestamps.modified_at,
                int(metadata.permissions.readable),
                int(metadata.permissions.writable),
                int(metadata.permissions.executable),
                metadata.owner,
                metadata.file_type.value,
                metadata.description,
                json.dumps(metadata.tags),
                self._serialize_embedding(metadata.embedding),
                self._serialize_chunks(metadata.chunks),
                metadata.size_bytes,
                metadata.line_count,
                json.dumps(metadata.extra),
            ),
        )
        self.conn.commit()

    def get_metadata(self, path: str) -> NodeMetadata | None:
        """Get metadata for a specific path."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM node_metadata WHERE path = ?", (path,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_metadata(row)

    def delete_metadata(self, path: str) -> bool:
        """Delete metadata for a specific path."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM node_metadata WHERE path = ?", (path,))
        self.conn.commit()
        return cursor.rowcount > 0

    def list_metadata(self, parent_path: str) -> list[NodeMetadata]:
        """List all metadata entries under a parent path."""
        assert self.conn is not None

        # Normalize parent path
        if parent_path == "/":
            pattern = "/%"
            exclude_pattern = "/%/%"
        else:
            parent_path = parent_path.rstrip("/")
            pattern = f"{parent_path}/%"
            exclude_pattern = f"{parent_path}/%/%"

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM node_metadata 
            WHERE path LIKE ? AND path NOT LIKE ?
            ORDER BY node_type, path
            """,
            (pattern, exclude_pattern),
        )

        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def move_metadata(self, old_path: str, new_path: str) -> bool:
        """Move metadata from old path to new path."""
        assert self.conn is not None

        cursor = self.conn.cursor()

        # Update the exact path
        cursor.execute(
            "UPDATE node_metadata SET path = ? WHERE path = ?",
            (new_path, old_path),
        )

        # Update all children paths
        old_prefix = old_path.rstrip("/") + "/"
        new_prefix = new_path.rstrip("/") + "/"

        cursor.execute(
            """
            UPDATE node_metadata 
            SET path = ? || SUBSTR(path, ?)
            WHERE path LIKE ?
            """,
            (new_prefix, len(old_prefix) + 1, old_prefix + "%"),
        )

        self.conn.commit()
        return True

    def search_by_tags(self, tags: list[str]) -> list[NodeMetadata]:
        """Search metadata by tags."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        # Simple tag search - check if any tag is present
        results = []
        cursor.execute("SELECT * FROM node_metadata")

        for row in cursor.fetchall():
            row_tags = json.loads(row["tags"])
            if any(tag in row_tags for tag in tags):
                results.append(self._row_to_metadata(row))

        return results

    def save_log(self, log: MemFSLog) -> None:
        """Save a log entry."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO logs (timestamp, operation, path, agent_id, details, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.timestamp,
                log.operation,
                log.path,
                log.agent_id,
                json.dumps(log.details),
                int(log.success),
                log.error_message,
            ),
        )
        self.conn.commit()

    def get_logs(
        self, limit: int = 100, offset: int = 0, path_filter: str | None = None
    ) -> list[MemFSLog]:
        """Get log entries with optional path filter."""
        assert self.conn is not None

        cursor = self.conn.cursor()

        if path_filter:
            cursor.execute(
                """
                SELECT * FROM logs 
                WHERE path LIKE ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (f"%{path_filter}%", limit, offset),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM logs 
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

        return [
            MemFSLog(
                timestamp=row["timestamp"],
                operation=row["operation"],
                path=row["path"],
                agent_id=row["agent_id"],
                details=json.loads(row["details"]),
                success=bool(row["success"]),
                error_message=row["error_message"],
            )
            for row in cursor.fetchall()
        ]

    def save_snapshot(self, snapshot: VersionSnapshot) -> None:
        """Save a version snapshot."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO version_snapshots 
            (version_id, timestamp, description, changed_paths, agent_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.version_id,
                snapshot.timestamp,
                snapshot.description,
                json.dumps(snapshot.changed_paths),
                snapshot.agent_id,
            ),
        )
        self.conn.commit()

    def get_snapshot(self, version_id: str) -> VersionSnapshot | None:
        """Get a specific version snapshot."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM version_snapshots WHERE version_id = ?", (version_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return VersionSnapshot(
            version_id=row["version_id"],
            timestamp=row["timestamp"],
            description=row["description"],
            changed_paths=json.loads(row["changed_paths"]),
            agent_id=row["agent_id"],
        )

    def list_snapshots(self, limit: int = 50) -> list[VersionSnapshot]:
        """List recent version snapshots."""
        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM version_snapshots 
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        return [
            VersionSnapshot(
                version_id=row["version_id"],
                timestamp=row["timestamp"],
                description=row["description"],
                changed_paths=json.loads(row["changed_paths"]),
                agent_id=row["agent_id"],
            )
            for row in cursor.fetchall()
        ]
