"""
MemFS - The core filesystem-like memory architecture for MemLearn.

MemFS provides a filesystem abstraction for LLM agents to store, manage,
and retrieve memory with semantic search capabilities.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Any

from memlearn.config import MemLearnConfig
from memlearn.databases.base import BaseDatabase
from memlearn.databases.sqlite_db import SQLiteDatabase
from memlearn.embedders.base import BaseEmbedder
from memlearn.embedders.openai_embedder import OpenAIEmbedder
from memlearn.rerankers.base import BaseReranker, NoOpReranker
from memlearn.rerankers.cohere_reranker import CohereReranker
from memlearn.sandboxes.base import BaseSandbox
from memlearn.sandboxes.local_sandbox import LocalSandbox
from memlearn.types import (
    Chunk,
    FileType,
    MemFSLog,
    NodeMetadata,
    NodeType,
    Permissions,
    SearchResult,
    Timestamps,
    ToolResult,
    VersionSnapshot,
)
from memlearn.vector_stores.base import BaseVectorStore
from memlearn.vector_stores.chroma_vdb import ChromaVectorStore

# Default AGENTS.md content that explains how to use MemFS
AGENTS_MD_CONTENT = """# MemFS Agent Guide

Welcome to MemFS - your filesystem-based memory system.

## Directory Structure

- `/raw/` - Read-only folder for raw data (conversation history, artifacts, skills)
  - `/raw/conversation-history/` - Past conversation histories
  - `/raw/artifacts/` - Files uploaded by the developer
  - `/raw/skills/` - Loaded skill definitions
- `/memory/` - Your main long-term memory folder (read/write)
  - Store insights, observations, learnings, user preferences here
  - This persists across sessions
- `/tmp/` - Temporary working memory (read/write)
  - Use for scratchpad, plans, temporary notes
  - Cleared at end of session
- `/mnt/` - Mounted shared folders (read/write with caution)
  - Shared across agents/users/organizations

## Available Operations

- **read**: Read file contents (with line numbers)
- **create**: Create new files or folders
- **edit**: Edit file contents (string replacement)
- **delete**: Delete files or folders
- **list**: List directory contents with metadata
- **move**: Move/rename files or folders
- **search**: Semantic + keyword hybrid search
- **peek**: View file/folder metadata without reading full content

## Best Practices

1. Organize memory logically with descriptive folder names
2. Add meaningful descriptions when creating files
3. Use tags for easy filtering
4. Use /tmp/ for working memory during tasks
5. Commit important insights to /memory/ for persistence
"""


class MemFS:
    """
    MemFS - Filesystem-like memory architecture for LLM agents.

    Provides a sandboxed filesystem with semantic search, metadata management,
    and version control capabilities designed for agentic memory and learning.
    """

    def __init__(
        self,
        config: MemLearnConfig | None = None,
        sandbox: BaseSandbox | None = None,
        database: BaseDatabase | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: BaseVectorStore | None = None,
        reranker: BaseReranker | None = None,
    ):
        """
        Initialize MemFS.

        Args:
            config: MemLearn configuration. Uses defaults if not provided.
            sandbox: Custom sandbox implementation.
            database: Custom database implementation.
            embedder: Custom embedder implementation.
            vector_store: Custom vector store implementation.
            reranker: Custom reranker implementation.
        """
        self.config = config or MemLearnConfig.from_env()

        # Initialize components
        self.sandbox = sandbox or self._create_sandbox()
        self.database = database or self._create_database()
        self.embedder = embedder or self._create_embedder()
        self.vector_store = vector_store or self._create_vector_store()
        self.reranker = reranker or self._create_reranker()

        self._initialized = False
        self._version_counter = 0

    def _create_sandbox(self) -> BaseSandbox:
        """Create sandbox based on config."""
        return LocalSandbox(prefix=self.config.sandbox.temp_dir_prefix)

    def _create_database(self) -> BaseDatabase:
        """Create database based on config."""
        return SQLiteDatabase(db_path=self.config.database.sqlite_path)

    def _create_embedder(self) -> BaseEmbedder:
        """Create embedder based on config."""
        return OpenAIEmbedder(
            api_key=self.config.embedder.api_key,
            model=self.config.embedder.model,
            dimensions=self.config.embedder.dimensions,
        )

    def _create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on config."""
        return ChromaVectorStore(
            persist_directory=self.config.vector_store.chroma_path,
            collection_name=self.config.vector_store.chroma_collection_name,
        )

    def _create_reranker(self) -> BaseReranker:
        """Create reranker based on config."""
        if self.config.reranker.provider == "none":
            return NoOpReranker()
        
        if self.config.reranker.api_key:
            return CohereReranker(
                api_key=self.config.reranker.api_key,
                model=self.config.reranker.model,
            )
        return NoOpReranker()

    def initialize(self) -> None:
        """Initialize MemFS and create default folder structure."""
        if self._initialized:
            return

        # Initialize components
        self.sandbox.initialize()
        self.database.initialize()
        self.vector_store.initialize()

        # Create default folder structure
        self._create_default_structure()
        self._initialized = True

    def _create_default_structure(self) -> None:
        """Create the default MemFS folder structure."""
        # Create root directories
        root_dirs = [
            ("raw", Permissions(readable=True, writable=False, executable=False)),
            ("raw/conversation-history", Permissions(readable=True, writable=False, executable=False)),
            ("raw/artifacts", Permissions(readable=True, writable=False, executable=False)),
            ("raw/skills", Permissions(readable=True, writable=False, executable=False)),
            ("memory", Permissions(readable=True, writable=True, executable=False)),
            ("tmp", Permissions(readable=True, writable=True, executable=False)),
            ("mnt", Permissions(readable=True, writable=True, executable=False)),
        ]

        for dir_path, permissions in root_dirs:
            self.sandbox.create_directory(dir_path)
            metadata = NodeMetadata(
                path=f"/{dir_path}",
                node_type=NodeType.FOLDER,
                permissions=permissions,
                owner="system",
                description=self._get_default_description(dir_path),
            )
            self.database.save_metadata(metadata)

        # Create AGENTS.md
        self.sandbox.write_file("AGENTS.md", AGENTS_MD_CONTENT)
        agents_md_metadata = NodeMetadata(
            path="/AGENTS.md",
            node_type=NodeType.FILE,
            permissions=Permissions(readable=True, writable=False, executable=False),
            owner="system",
            file_type=FileType.MARKDOWN,
            description="Comprehensive guide for agents on how to use MemFS",
            size_bytes=len(AGENTS_MD_CONTENT.encode()),
            line_count=AGENTS_MD_CONTENT.count("\n") + 1,
        )
        self.database.save_metadata(agents_md_metadata)

    def _get_default_description(self, path: str) -> str:
        """Get default description for system directories."""
        descriptions = {
            "raw": "Read-only folder for raw data including conversation history and artifacts",
            "raw/conversation-history": "Stores all past conversation histories",
            "raw/artifacts": "Files uploaded manually by the developer",
            "raw/skills": "Skill definitions loaded by the developer",
            "memory": "Main long-term memory folder - agent has full control",
            "tmp": "Temporary working memory - cleared at end of session",
            "mnt": "Mounted shared folders for multi-agent collaboration",
        }
        return descriptions.get(path, "")

    def close(self) -> None:
        """Close MemFS and clean up resources."""
        if not self._initialized:
            return

        self.vector_store.close()
        self.database.close()
        self.sandbox.cleanup()
        self._initialized = False

    def __enter__(self) -> "MemFS":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # File Operations
    # =========================================================================

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        include_line_numbers: bool | None = None,
    ) -> ToolResult:
        """
        Read file contents.

        Args:
            path: Path to the file.
            start_line: Starting line number (1-indexed). None for beginning.
            end_line: Ending line number (inclusive). None for end.
            include_line_numbers: Whether to include line numbers in output.

        Returns:
            ToolResult with file contents.
        """
        path = self._normalize_path(path)

        # Check if file exists
        if not self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"File not found: {path}. Please check the path and try again.",
            )

        if self.sandbox.is_dir(path):
            return ToolResult(
                status="error",
                message=f"Path is a directory, not a file: {path}. Use 'list' to view directory contents.",
            )

        # Check permissions
        metadata = self.database.get_metadata(path)
        if metadata and not metadata.permissions.readable:
            return ToolResult(
                status="error",
                message=f"Permission denied: {path} is not readable.",
            )

        try:
            content = self.sandbox.read_file(path)
            lines = content.split("\n")
            total_lines = len(lines)

            # Apply line range
            start = (start_line or 1) - 1  # Convert to 0-indexed
            end = end_line or total_lines
            
            # Clamp to valid range
            start = max(0, min(start, total_lines))
            end = max(start, min(end, total_lines))
            
            selected_lines = lines[start:end]

            # Format output
            if include_line_numbers is None:
                include_line_numbers = self.config.agent.include_line_numbers

            if include_line_numbers:
                width = len(str(end))
                formatted_lines = [
                    f"{i + start + 1:>{width}}| {line}"
                    for i, line in enumerate(selected_lines)
                ]
                output = "\n".join(formatted_lines)
            else:
                output = "\n".join(selected_lines)

            # Update access time
            if metadata:
                metadata.timestamps.touch_access()
                self.database.save_metadata(metadata)

            # Log operation
            self._log_operation("read_file", path, {"lines": f"{start+1}-{end}"})

            return ToolResult(
                status="success",
                message=f"Read {path} (lines {start+1}-{end} of {total_lines})",
                data={"content": output, "total_lines": total_lines},
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to read file: {str(e)}",
            )

    def create_file(
        self,
        path: str,
        content: str = "",
        description: str = "",
        tags: list[str] | None = None,
    ) -> ToolResult:
        """
        Create a new file.

        Args:
            path: Path for the new file.
            content: Initial content.
            description: Description of the file.
            tags: Optional tags for the file.

        Returns:
            ToolResult indicating success or failure.
        """
        path = self._normalize_path(path)

        # Check if path already exists
        if self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"Path already exists: {path}. Use 'edit' to modify existing files.",
            )

        # Check parent directory permissions
        parent_path = self._get_parent_path(path)
        if parent_path:
            parent_meta = self.database.get_metadata(parent_path)
            if parent_meta and not parent_meta.permissions.writable:
                return ToolResult(
                    status="error",
                    message=f"Permission denied: Cannot create files in {parent_path}.",
                )

        try:
            # Write file
            self.sandbox.write_file(path, content)

            # Detect file type
            file_type = self._detect_file_type(path, content)

            # Create metadata
            metadata = NodeMetadata(
                path=path,
                node_type=NodeType.FILE,
                owner=self.config.agent_id,
                file_type=file_type,
                description=description,
                tags=tags or [],
                size_bytes=len(content.encode()),
                line_count=content.count("\n") + 1 if content else 0,
            )

            # Generate embeddings if auto_embed is enabled
            if self.config.auto_embed and content:
                self._embed_file(metadata, content)

            self.database.save_metadata(metadata)

            # Log and version
            self._log_operation("create_file", path, {"size": metadata.size_bytes})
            self._create_version_snapshot(f"Created file: {path}", [path])

            return ToolResult(
                status="success",
                message=f"Created file: {path}",
                data={"path": path, "size": metadata.size_bytes},
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to create file: {str(e)}",
            )

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        description: str | None = None,
    ) -> ToolResult:
        """
        Edit a file using string replacement.

        Args:
            path: Path to the file.
            old_string: String to replace (must be unique in file).
            new_string: Replacement string.
            description: Optional updated description.

        Returns:
            ToolResult indicating success or failure.
        """
        path = self._normalize_path(path)

        # Check if file exists
        if not self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"File not found: {path}",
            )

        if self.sandbox.is_dir(path):
            return ToolResult(
                status="error",
                message=f"Path is a directory: {path}. Cannot edit directories.",
            )

        # Check permissions
        metadata = self.database.get_metadata(path)
        if metadata and not metadata.permissions.writable:
            return ToolResult(
                status="error",
                message=f"Permission denied: {path} is not writable.",
            )

        try:
            content = self.sandbox.read_file(path)

            # Check uniqueness of old_string
            count = content.count(old_string)
            if count == 0:
                return ToolResult(
                    status="error",
                    message=f"String not found in file. The exact string to replace was not found in {path}. "
                            "Make sure you're using the exact text including whitespace.",
                )
            if count > 1:
                return ToolResult(
                    status="error",
                    message=f"String appears {count} times in file. Please provide a more unique string "
                            "that includes surrounding context to ensure only one match.",
                )

            # Perform replacement
            new_content = content.replace(old_string, new_string, 1)
            self.sandbox.write_file(path, new_content)

            # Update metadata
            if metadata:
                metadata.timestamps.touch_modify()
                metadata.size_bytes = len(new_content.encode())
                metadata.line_count = new_content.count("\n") + 1
                if description:
                    metadata.description = description

                # Re-embed if auto_embed enabled
                if self.config.auto_embed:
                    self._embed_file(metadata, new_content)

                self.database.save_metadata(metadata)

            # Log and version
            self._log_operation("edit_file", path, {
                "old_length": len(old_string),
                "new_length": len(new_string),
            })
            self._create_version_snapshot(f"Edited file: {path}", [path])

            return ToolResult(
                status="success",
                message=f"Successfully edited {path}",
                data={"path": path, "size": len(new_content.encode())},
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to edit file: {str(e)}",
            )

    def delete(self, path: str, recursive: bool = False) -> ToolResult:
        """
        Delete a file or directory.

        Args:
            path: Path to delete.
            recursive: If True, delete directories recursively.

        Returns:
            ToolResult indicating success or failure.
        """
        path = self._normalize_path(path)

        # Prevent deletion of root directories
        protected_paths = ["/raw", "/memory", "/tmp", "/mnt", "/AGENTS.md"]
        if path in protected_paths or path.startswith("/raw/"):
            return ToolResult(
                status="error",
                message=f"Cannot delete protected path: {path}",
            )

        # Check if path exists
        if not self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"Path not found: {path}",
            )

        # Check permissions
        metadata = self.database.get_metadata(path)
        if metadata and not metadata.permissions.writable:
            return ToolResult(
                status="error",
                message=f"Permission denied: {path} is not writable.",
            )

        try:
            is_dir = self.sandbox.is_dir(path)

            if is_dir:
                self.sandbox.delete_directory(path, recursive=recursive)
            else:
                self.sandbox.delete_file(path)

            # Delete metadata (and children if directory)
            self.database.delete_metadata(path)
            if is_dir:
                # Delete child metadata
                for child in self.database.list_metadata(path):
                    self.database.delete_metadata(child.path)

            # Delete from vector store
            self._delete_embeddings(path)

            # Log and version
            self._log_operation("delete", path, {"recursive": recursive})
            self._create_version_snapshot(f"Deleted: {path}", [path])

            return ToolResult(
                status="success",
                message=f"Deleted: {path}",
            )

        except OSError as e:
            if "not empty" in str(e).lower():
                return ToolResult(
                    status="error",
                    message=f"Directory not empty: {path}. Use recursive=True to delete non-empty directories.",
                )
            return ToolResult(
                status="error",
                message=f"Failed to delete: {str(e)}",
            )

    # =========================================================================
    # Directory Operations
    # =========================================================================

    def create_directory(
        self,
        path: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ToolResult:
        """
        Create a directory (and parent directories if needed).

        Args:
            path: Path for the new directory.
            description: Description of the directory.
            tags: Optional tags.

        Returns:
            ToolResult indicating success or failure.
        """
        path = self._normalize_path(path)

        # Check if path already exists
        if self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"Path already exists: {path}",
            )

        # Check parent permissions
        parent_path = self._get_parent_path(path)
        if parent_path:
            parent_meta = self.database.get_metadata(parent_path)
            if parent_meta and not parent_meta.permissions.writable:
                return ToolResult(
                    status="error",
                    message=f"Permission denied: Cannot create directories in {parent_path}.",
                )

        try:
            self.sandbox.create_directory(path)

            # Create metadata
            metadata = NodeMetadata(
                path=path,
                node_type=NodeType.FOLDER,
                owner=self.config.agent_id,
                description=description,
                tags=tags or [],
            )

            # Embed description if provided
            if self.config.auto_embed and description:
                metadata.embedding = self.embedder.embed(description)
                self._add_to_vector_store(path, metadata.embedding, {
                    "path": path,
                    "type": "folder",
                    "description": description,
                })

            self.database.save_metadata(metadata)

            # Log and version
            self._log_operation("create_directory", path)
            self._create_version_snapshot(f"Created directory: {path}", [path])

            return ToolResult(
                status="success",
                message=f"Created directory: {path}",
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to create directory: {str(e)}",
            )

    def list_directory(
        self,
        path: str = "/",
        show_hidden: bool = False,
        max_depth: int = 1,
    ) -> ToolResult:
        """
        List contents of a directory with metadata.

        Args:
            path: Path to list.
            show_hidden: Whether to show hidden files.
            max_depth: Maximum depth to recurse (1 = current dir only).

        Returns:
            ToolResult with directory listing.
        """
        path = self._normalize_path(path)

        if not self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"Directory not found: {path}",
            )

        if not self.sandbox.is_dir(path):
            return ToolResult(
                status="error",
                message=f"Path is not a directory: {path}",
            )

        try:
            entries = self._list_directory_recursive(path, max_depth, show_hidden)

            # Format output
            output_lines = [f"Contents of {path}:"]
            output_lines.append("")

            for entry in entries:
                meta = self.database.get_metadata(entry["path"])
                
                if entry["is_dir"]:
                    type_indicator = "📁"
                    size_info = f"{entry.get('child_count', 0)} items"
                else:
                    type_indicator = "📄"
                    size_info = self._format_size(entry.get("size", 0))

                name = entry["name"]
                if entry["depth"] > 0:
                    name = "  " * entry["depth"] + name

                desc = ""
                if meta and meta.description:
                    desc = f" - {meta.description[:50]}{'...' if len(meta.description) > 50 else ''}"

                output_lines.append(f"{type_indicator} {name} ({size_info}){desc}")

            return ToolResult(
                status="success",
                message=f"Listed {len(entries)} items in {path}",
                data={"entries": entries, "formatted": "\n".join(output_lines)},
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to list directory: {str(e)}",
            )

    def _list_directory_recursive(
        self,
        path: str,
        max_depth: int,
        show_hidden: bool,
        current_depth: int = 0,
    ) -> list[dict]:
        """Recursively list directory contents."""
        entries = []
        
        try:
            items = self.sandbox.list_directory(path)
        except Exception:
            return entries

        for item in items:
            if not show_hidden and item.startswith("."):
                continue

            item_path = f"{path}/{item}" if path != "/" else f"/{item}"
            is_dir = self.sandbox.is_dir(item_path)

            entry = {
                "name": item,
                "path": item_path,
                "is_dir": is_dir,
                "depth": current_depth,
            }

            if is_dir:
                children = self.sandbox.list_directory(item_path)
                entry["child_count"] = len(children)
            else:
                entry["size"] = self.sandbox.get_size(item_path)

            entries.append(entry)

            # Recurse into directories
            if is_dir and current_depth < max_depth - 1:
                child_entries = self._list_directory_recursive(
                    item_path, max_depth, show_hidden, current_depth + 1
                )
                entries.extend(child_entries)

        return entries

    # =========================================================================
    # Move/Rename Operations
    # =========================================================================

    def move(self, src: str, dst: str) -> ToolResult:
        """
        Move or rename a file or directory.

        Args:
            src: Source path.
            dst: Destination path.

        Returns:
            ToolResult indicating success or failure.
        """
        src = self._normalize_path(src)
        dst = self._normalize_path(dst)

        # Check source exists
        if not self.sandbox.exists(src):
            return ToolResult(
                status="error",
                message=f"Source not found: {src}",
            )

        # Check source permissions
        src_meta = self.database.get_metadata(src)
        if src_meta and not src_meta.permissions.writable:
            return ToolResult(
                status="error",
                message=f"Permission denied: {src} is not writable.",
            )

        # Check destination parent permissions
        dst_parent = self._get_parent_path(dst)
        if dst_parent:
            dst_parent_meta = self.database.get_metadata(dst_parent)
            if dst_parent_meta and not dst_parent_meta.permissions.writable:
                return ToolResult(
                    status="error",
                    message=f"Permission denied: Cannot move to {dst_parent}.",
                )

        try:
            self.sandbox.move(src, dst)

            # Update metadata paths
            self.database.move_metadata(src, dst)

            # Update vector store
            self._move_embeddings(src, dst)

            # Log and version
            self._log_operation("move", src, {"destination": dst})
            self._create_version_snapshot(f"Moved {src} to {dst}", [src, dst])

            return ToolResult(
                status="success",
                message=f"Moved {src} to {dst}",
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to move: {str(e)}",
            )

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query: str,
        path: str = "/",
        search_type: str = "hybrid",
        top_k: int = 10,
        tags: list[str] | None = None,
    ) -> ToolResult:
        """
        Search MemFS using semantic and/or keyword search.

        Args:
            query: Search query.
            path: Base path to search from.
            search_type: One of "semantic", "keyword", "hybrid".
            top_k: Maximum number of results.
            tags: Optional tag filter.

        Returns:
            ToolResult with search results.
        """
        path = self._normalize_path(path)

        try:
            results: list[SearchResult] = []

            # Semantic search
            if search_type in ("semantic", "hybrid"):
                semantic_results = self._semantic_search(query, path, top_k * 2, tags)
                results.extend(semantic_results)

            # Keyword search
            if search_type in ("keyword", "hybrid"):
                keyword_results = self._keyword_search(query, path, top_k * 2)
                results.extend(keyword_results)

            # Deduplicate and sort by score
            seen_paths = {}
            for result in results:
                key = f"{result.path}:{result.chunk_index or 0}"
                if key not in seen_paths or result.score > seen_paths[key].score:
                    seen_paths[key] = result

            results = sorted(seen_paths.values(), key=lambda x: x.score, reverse=True)

            # Rerank if we have enough results
            if len(results) > 1 and search_type == "hybrid":
                results = self._rerank_results(query, results, top_k)
            else:
                results = results[:top_k]

            # Format output
            output_lines = [f"Search results for '{query}':"]
            output_lines.append("")

            for i, result in enumerate(results, 1):
                output_lines.append(f"{i}. {result.path} (score: {result.score:.3f})")
                if result.content_snippet:
                    snippet = result.content_snippet[:100]
                    if len(result.content_snippet) > 100:
                        snippet += "..."
                    output_lines.append(f"   {snippet}")
                output_lines.append("")

            return ToolResult(
                status="success",
                message=f"Found {len(results)} results for '{query}'",
                data={
                    "results": [
                        {
                            "path": r.path,
                            "score": r.score,
                            "snippet": r.content_snippet,
                            "start_line": r.start_line,
                            "end_line": r.end_line,
                        }
                        for r in results
                    ],
                    "formatted": "\n".join(output_lines),
                },
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Search failed: {str(e)}",
            )

    def _semantic_search(
        self,
        query: str,
        path: str,
        top_k: int,
        tags: list[str] | None,
    ) -> list[SearchResult]:
        """Perform semantic search using embeddings."""
        query_embedding = self.embedder.embed(query)

        filter_metadata = None
        if path != "/":
            filter_metadata = {"path_prefix": path}

        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        results = []
        for vr in vector_results:
            # Filter by tags if specified
            if tags:
                meta = self.database.get_metadata(vr.metadata.get("path", ""))
                if meta and not any(t in meta.tags for t in tags):
                    continue

            results.append(SearchResult(
                path=vr.metadata.get("path", ""),
                score=vr.score,
                content_snippet=vr.metadata.get("content", ""),
                chunk_index=vr.metadata.get("chunk_index"),
                start_line=vr.metadata.get("start_line"),
                end_line=vr.metadata.get("end_line"),
            ))

        return results

    def _keyword_search(
        self,
        query: str,
        path: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Perform keyword search using regex."""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        def search_in_file(file_path: str) -> None:
            try:
                content = self.sandbox.read_file(file_path)
                content_lower = content.lower()

                # Simple scoring based on word matches
                score = 0
                for word in query_words:
                    count = content_lower.count(word)
                    score += count * 0.1

                if score > 0:
                    # Find best matching snippet
                    lines = content.split("\n")
                    best_line_idx = 0
                    best_line_score = 0

                    for i, line in enumerate(lines):
                        line_lower = line.lower()
                        line_score = sum(1 for w in query_words if w in line_lower)
                        if line_score > best_line_score:
                            best_line_score = line_score
                            best_line_idx = i

                    # Get context around best line
                    start = max(0, best_line_idx - 1)
                    end = min(len(lines), best_line_idx + 2)
                    snippet = "\n".join(lines[start:end])

                    results.append(SearchResult(
                        path=file_path,
                        score=min(score, 1.0),
                        content_snippet=snippet,
                        start_line=start + 1,
                        end_line=end,
                    ))

            except Exception:
                pass

        def search_recursive(dir_path: str) -> None:
            try:
                items = self.sandbox.list_directory(dir_path)
                for item in items:
                    item_path = f"{dir_path}/{item}" if dir_path != "/" else f"/{item}"
                    if self.sandbox.is_dir(item_path):
                        search_recursive(item_path)
                    else:
                        search_in_file(item_path)
            except Exception:
                pass

        search_recursive(path)

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank search results using the reranker."""
        if not results:
            return []

        documents = [
            f"{r.path}: {r.content_snippet or ''}" for r in results
        ]

        try:
            reranked = self.reranker.rerank(query, documents, top_n=top_k)

            reranked_results = []
            for rr in reranked:
                original = results[rr.index]
                reranked_results.append(SearchResult(
                    path=original.path,
                    score=rr.score,
                    content_snippet=original.content_snippet,
                    chunk_index=original.chunk_index,
                    start_line=original.start_line,
                    end_line=original.end_line,
                ))

            return reranked_results

        except Exception:
            # Fall back to original ordering
            return results[:top_k]

    # =========================================================================
    # Peek Operation
    # =========================================================================

    def peek(self, path: str) -> ToolResult:
        """
        View metadata about a file or folder without reading full content.

        Args:
            path: Path to peek at.

        Returns:
            ToolResult with metadata information.
        """
        path = self._normalize_path(path)

        if not self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"Path not found: {path}",
            )

        metadata = self.database.get_metadata(path)
        if not metadata:
            # Create basic metadata from filesystem
            is_dir = self.sandbox.is_dir(path)
            metadata = NodeMetadata(
                path=path,
                node_type=NodeType.FOLDER if is_dir else NodeType.FILE,
            )

        # Format output
        output_lines = [f"Metadata for {path}:"]
        output_lines.append(f"  Type: {'folder' if metadata.is_folder else 'file'}")
        output_lines.append(f"  Owner: {metadata.owner}")
        output_lines.append(f"  Description: {metadata.description or '(none)'}")
        output_lines.append(f"  Tags: {', '.join(metadata.tags) if metadata.tags else '(none)'}")

        if metadata.is_file:
            output_lines.append(f"  File type: {metadata.file_type.value}")
            output_lines.append(f"  Size: {self._format_size(metadata.size_bytes)}")
            output_lines.append(f"  Lines: {metadata.line_count}")

        perms = metadata.permissions
        perm_str = f"{'r' if perms.readable else '-'}{'w' if perms.writable else '-'}{'x' if perms.executable else '-'}"
        output_lines.append(f"  Permissions: {perm_str}")

        ts = metadata.timestamps
        output_lines.append(f"  Created: {self._format_timestamp(ts.created_at)}")
        output_lines.append(f"  Modified: {self._format_timestamp(ts.modified_at)}")
        output_lines.append(f"  Accessed: {self._format_timestamp(ts.accessed_at)}")

        return ToolResult(
            status="success",
            message=f"Metadata for {path}",
            data={
                "path": path,
                "type": metadata.node_type.value,
                "description": metadata.description,
                "tags": metadata.tags,
                "size_bytes": metadata.size_bytes,
                "line_count": metadata.line_count,
                "permissions": {
                    "readable": perms.readable,
                    "writable": perms.writable,
                    "executable": perms.executable,
                },
                "formatted": "\n".join(output_lines),
            },
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _normalize_path(self, path: str) -> str:
        """Normalize a path to absolute form."""
        if not path.startswith("/"):
            path = "/" + path
        # Remove trailing slash except for root
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        return path

    def _get_parent_path(self, path: str) -> str | None:
        """Get parent directory path."""
        if path == "/":
            return None
        parts = path.rsplit("/", 1)
        return parts[0] if parts[0] else "/"

    def _detect_file_type(self, path: str, content: str) -> FileType:
        """Detect file type from extension and content."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

        type_map = {
            "md": FileType.MARKDOWN,
            "markdown": FileType.MARKDOWN,
            "json": FileType.JSON,
            "py": FileType.CODE,
            "js": FileType.CODE,
            "ts": FileType.CODE,
            "jsx": FileType.CODE,
            "tsx": FileType.CODE,
            "java": FileType.CODE,
            "c": FileType.CODE,
            "cpp": FileType.CODE,
            "go": FileType.CODE,
            "rs": FileType.CODE,
            "rb": FileType.CODE,
            "php": FileType.CODE,
            "png": FileType.IMAGE,
            "jpg": FileType.IMAGE,
            "jpeg": FileType.IMAGE,
            "gif": FileType.IMAGE,
            "pdf": FileType.PDF,
        }

        return type_map.get(ext, FileType.TEXT)

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _format_timestamp(self, ts: float) -> str:
        """Format timestamp to human readable."""
        import datetime
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _embed_file(self, metadata: NodeMetadata, content: str) -> None:
        """Generate embeddings for file content."""
        # Embed description
        if metadata.description:
            metadata.embedding = self.embedder.embed(metadata.description)

        # Chunk and embed content
        chunks = self.embedder.chunk_text(
            content,
            chunk_size=self.config.embedder.chunk_size,
            overlap=self.config.embedder.chunk_overlap,
        )

        metadata.chunks = []
        chunk_texts = []
        chunk_metadata = []

        for i, (chunk_text, start_line, end_line) in enumerate(chunks):
            chunk = Chunk.from_content(chunk_text, start_line, end_line)
            metadata.chunks.append(chunk)
            chunk_texts.append(chunk_text)
            chunk_metadata.append({
                "path": metadata.path,
                "chunk_index": i,
                "start_line": start_line,
                "end_line": end_line,
                "content": chunk_text[:200],  # Store snippet for search results
            })

        # Batch embed chunks
        if chunk_texts:
            embeddings = self.embedder.embed_batch(chunk_texts)
            for i, embedding in enumerate(embeddings):
                metadata.chunks[i].embedding = embedding

            # Add to vector store
            ids = [f"{metadata.path}:chunk:{i}" for i in range(len(chunks))]
            self.vector_store.add(ids, embeddings, chunk_metadata, chunk_texts)

        # Add description embedding to vector store
        if metadata.embedding:
            self.vector_store.add(
                [f"{metadata.path}:description"],
                [metadata.embedding],
                [{"path": metadata.path, "type": "description", "content": metadata.description}],
                [metadata.description],
            )

    def _add_to_vector_store(
        self,
        path: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Add a single embedding to the vector store."""
        self.vector_store.add(
            [f"{path}:description"],
            [embedding],
            [metadata],
        )

    def _delete_embeddings(self, path: str) -> None:
        """Delete all embeddings for a path."""
        try:
            # Get metadata to find chunk count
            metadata = self.database.get_metadata(path)
            ids_to_delete = [f"{path}:description"]

            if metadata and metadata.chunks:
                for i in range(len(metadata.chunks)):
                    ids_to_delete.append(f"{path}:chunk:{i}")

            self.vector_store.delete(ids_to_delete)
        except Exception:
            pass

    def _move_embeddings(self, old_path: str, new_path: str) -> None:
        """Move embeddings when a path is renamed."""
        # For simplicity, we delete and re-add
        # A production implementation might update in place
        try:
            metadata = self.database.get_metadata(new_path)
            if metadata and metadata.is_file:
                content = self.sandbox.read_file(new_path)
                self._delete_embeddings(old_path)
                self._embed_file(metadata, content)
                self.database.save_metadata(metadata)
        except Exception:
            pass

    def _log_operation(
        self,
        operation: str,
        path: str,
        details: dict | None = None,
    ) -> None:
        """Log a MemFS operation."""
        if not self.config.auto_log:
            return

        log = MemFSLog.create(
            operation=operation,
            path=path,
            agent_id=self.config.agent_id,
            details=details or {},
        )
        self.database.save_log(log)

    def _create_version_snapshot(
        self,
        description: str,
        changed_paths: list[str],
    ) -> None:
        """Create a version snapshot for undo support."""
        self._version_counter += 1
        snapshot = VersionSnapshot(
            version_id=f"v{self._version_counter}_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            description=description,
            changed_paths=changed_paths,
            agent_id=self.config.agent_id,
        )
        self.database.save_snapshot(snapshot)


# =========================================================================
# Factory Functions
# =========================================================================

def create_memfs(config: MemLearnConfig | None = None) -> MemFS:
    """
    Create and initialize a new MemFS instance.

    Args:
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        Initialized MemFS instance.
    """
    memfs = MemFS(config=config)
    memfs.initialize()
    return memfs


def load_memfs(
    agent_id: str | None = None,
    config: MemLearnConfig | None = None,
) -> MemFS:
    """
    Load a MemFS instance, potentially from persistent storage.

    Args:
        agent_id: Agent ID for loading specific memory.
        config: Optional configuration.

    Returns:
        MemFS instance.
    """
    if config is None:
        config = MemLearnConfig.from_env()

    if agent_id:
        config.agent_id = agent_id

    return create_memfs(config)
