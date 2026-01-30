"""
MemFS - The core filesystem-like memory architecture for MemLearn.

MemFS provides a filesystem abstraction for LLM agents to store, manage,
and retrieve memory with semantic search capabilities.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from memlearn.config import MemLearnConfig
from memlearn.databases.base import BaseDatabase
from memlearn.databases.sqlite_db import SQLiteDatabase
from memlearn.embedders.base import BaseEmbedder
from memlearn.embedders.openai_embedder import OpenAIEmbedder
from memlearn.llms.base import BaseLLM
from memlearn.llms.openai_llm import OpenAILLM
from memlearn.rerankers.base import BaseReranker, NoOpReranker
from memlearn.rerankers.cohere_reranker import CohereReranker
from memlearn.sandboxes.base import BaseSandbox
from memlearn.sandboxes.local_sandbox import LocalSandbox
from memlearn.types import (
    Agent,
    Chunk,
    ConversationHistory,
    FileType,
    MemFSLog,
    MountInfo,
    MountSourceType,
    NodeMetadata,
    NodeType,
    Permissions,
    SearchResult,
    Session,
    SessionStatus,
    Timestamps,
    ToolResult,
    VersionSnapshot,
)
from memlearn.vector_stores.base import BaseVectorStore
from memlearn.vector_stores.chroma_vdb import ChromaVectorStore
from memlearn.embedders.conversation_chunker import (
    chunk_conversation_history,
    create_chunk_for_embedding,
)
from memlearn.prompts import get_memory_note_prompts

# Folders to sync between persistent and ephemeral storage
PERSISTENT_FOLDERS = ["memory", "raw", "mnt"]

# Conversation history file path (markdown format for agent-friendly access)
CONVERSATION_HISTORY_PATH = "/raw/conversation-history/CURRENT.md"

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
- **find**: Find files by name pattern (glob, regex, or exact match)
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
        llm: BaseLLM | None = None,
        read_only: bool = False,
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
            llm: Custom LLM implementation for summarization and reflection.
            read_only: If True, the MemFS instance operates in read-only mode.
                This disables:
                - Summarization on spindown
                - Conversation history storage
                - Memory note updates on spindown
                - Write operations via tool providers (create, edit, delete, mkdir, move)
                - Persistent state sync on spindown
        """
        self.config = config or MemLearnConfig.from_env()
        self.read_only = read_only

        # Initialize components
        self.sandbox = sandbox or self._create_sandbox()
        self.database = database or self._create_database()
        self.embedder = embedder or self._create_embedder()
        self.vector_store = vector_store or self._create_vector_store()
        self.reranker = reranker or self._create_reranker()
        self.llm = llm or self._create_llm()

        self._initialized = False
        self._version_counter = 0
        self._session_branch_name: str | None = None

    def _debug_log(self, message: str) -> None:
        """Log a debug message if debug mode is enabled."""
        if self.config.debug:
            print(f"[MemLearn DEBUG] {message}")

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

    def _create_llm(self) -> BaseLLM | None:
        """Create LLM based on config."""
        # Only create LLM if summarization OR memory note update is enabled
        needs_llm = (
            self.config.llm.summarize_on_spindown
            or self.config.llm.update_memory_note_on_spindown
        )
        if not needs_llm:
            self._debug_log(
                "_create_llm() returning None: neither summarization nor memory note update enabled"
            )
            return None

        if self.config.llm.provider == "openai":
            self._debug_log(
                f"_create_llm() creating OpenAILLM with model={self.config.llm.model}"
            )
            return OpenAILLM(
                api_key=self.config.llm.api_key,
                model=self.config.llm.model,
                default_max_tokens=self.config.llm.max_summary_tokens,
            )

        # Future: Add anthropic, gemini providers
        self._debug_log(
            f"_create_llm() returning None: unsupported provider {self.config.llm.provider}"
        )
        return None

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

        # Initialize git version control and commit initial state
        self._git_init()
        self._git_commit_all("init")

        self._initialized = True

    def _create_default_structure(self) -> None:
        """Create the default MemFS folder structure."""
        # Create root directories
        root_dirs = [
            ("raw", Permissions(readable=True, writable=False, executable=False)),
            (
                "raw/conversation-history",
                Permissions(readable=True, writable=False, executable=False),
            ),
            (
                "raw/artifacts",
                Permissions(readable=True, writable=False, executable=False),
            ),
            (
                "raw/skills",
                Permissions(readable=True, writable=False, executable=False),
            ),
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
        """Close MemFS and clean up resources without persisting state.

        Use spindown() instead if you want to persist state before closing.
        """
        if not self._initialized:
            return

        self.vector_store.close()
        self.database.close()
        self.sandbox.cleanup()
        self._initialized = False

    def __enter__(self) -> "MemFS":
        """Context manager entry."""
        # If already initialized via for_agent(), just return self
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # If we have an active session, spin down properly
        if self.config.session_id:
            status = SessionStatus.ABORTED if exc_type else SessionStatus.COMPLETED
            self.spindown(status=status)
        else:
            self.close()

    # =========================================================================
    # Session Lifecycle Methods
    # =========================================================================

    def spinup(self, agent_name: str, session_id: str | None = None) -> None:
        """
        Initialize MemFS for an agent session.

        This is the main entry point for starting a persistent agent session.
        It will:
        1. Load or create the agent entity in the database
        2. Create a new session record
        3. Initialize ephemeral sandbox
        4. Sync persistent storage into the sandbox
        5. Initialize conversation history

        Args:
            agent_name: Unique name for the agent (e.g., "code-editor")
            session_id: Optional custom session ID. Generated if not provided.
        """
        if self._initialized:
            raise RuntimeError("MemFS is already initialized. Call spindown() first.")

        # Initialize database first (needed for agent operations)
        self.database.initialize()

        # Load or create agent
        agent = self.database.get_agent_by_name(agent_name)
        if agent is None:
            agent = Agent.create(name=agent_name)
            self.database.create_agent(agent)

        # Check for any orphaned active sessions and end them
        active_session = self.database.get_active_session_for_agent(agent.agent_id)
        if active_session:
            self.database.end_session(active_session.session_id, SessionStatus.ABORTED)

        # Create new session
        session = Session.create(agent_id=agent.agent_id)
        if session_id:
            session.session_id = session_id
        self.database.create_session(session)

        # Update config with agent and session info
        self.config.agent_id = agent.agent_id
        self.config.session_id = session.session_id
        self.config.agent_name = agent.name

        # Initialize sandbox
        self.sandbox.initialize()
        self.vector_store.initialize()

        # Create default folder structure in sandbox
        self._create_default_structure()

        # Sync from persistent storage to sandbox
        persistent_path = self.config.sandbox.get_agent_persistent_path(agent.agent_id)
        if persistent_path.exists():
            self.sandbox.sync_from_persistent(str(persistent_path), PERSISTENT_FOLDERS)
            # Clear /tmp after sync (it should always start empty)
            self.sandbox.clear_directory_contents("tmp")

        # Remount any previously mounted folders
        self._remount_saved_mounts()

        # Initialize conversation history for this session
        self._init_conversation_history()

        # Initialize git version control, commit initial state, and create session branch
        self._git_init()
        self._git_commit_all("init")
        self._git_create_session_branch()

        self._initialized = True

    def spindown(self, status: SessionStatus = SessionStatus.COMPLETED) -> None:
        """
        End session and persist state.

        This will (unless in read_only mode):
        1. Summarize conversation history using LLM (if enabled)
        2. Archive conversation history (rename CURRENT to timestamp)
        3. Generate/update memory note for next session (if enabled)
        4. Sync sandbox to persistent storage
        5. Update session record
        6. Clean up ephemeral sandbox

        In read_only mode, only cleanup is performed - no state modifications.

        Args:
            status: Final session status (completed or aborted)
        """
        self._debug_log(
            f"spindown() called with status={status}, read_only={self.read_only}"
        )
        if not self._initialized:
            self._debug_log("spindown() skipped: not initialized")
            return

        self._debug_log(
            f"agent_id={self.config.agent_id}, session_id={self.config.session_id}"
        )

        # In read-only mode, skip all write operations
        if not self.read_only:
            # Generate conversation summary using LLM (if enabled and conversation exists)
            conversation_summary = None
            if self.config.llm.summarize_on_spindown and self.llm is not None:
                self._debug_log("Generating conversation summary...")
                conversation_summary = self._summarize_conversation()
                self._debug_log(
                    f"Conversation summary generated: {conversation_summary is not None}"
                )

            # Archive conversation history (with summary in metadata)
            self._debug_log("Archiving conversation history...")
            self._archive_conversation_history(summary=conversation_summary)

            # Generate/update memory note for next session
            self._debug_log(
                f"Memory note update check: update_memory_note_on_spindown={self.config.llm.update_memory_note_on_spindown}, llm={self.llm is not None}"
            )
            if self.config.llm.update_memory_note_on_spindown and self.llm is not None:
                self._debug_log("Generating memory note...")
                new_note = self._generate_memory_note()
                self._debug_log(
                    f"Memory note generated: {new_note is not None}, content length: {len(new_note) if new_note else 0}"
                )
                if new_note:
                    self._debug_log(
                        f"New memory note preview: {new_note[:200]}..."
                        if len(new_note) > 200
                        else f"New memory note: {new_note}"
                    )
                    self._update_memory_note(new_note)
                else:
                    self._debug_log(
                        "Memory note generation returned None - note will NOT be updated"
                    )

            # Commit session end with summary (git version control)
            self._git_session_end_commit(summary=conversation_summary)

            # Sync to persistent storage (excluding /tmp)
            if self.config.agent_id:
                persistent_path = self.config.sandbox.get_agent_persistent_path(
                    self.config.agent_id
                )
                self.sandbox.sync_to_persistent(
                    str(persistent_path), PERSISTENT_FOLDERS
                )

            # End session in database
            if self.config.session_id:
                self.database.end_session(self.config.session_id, status)
        else:
            self._debug_log("Read-only mode: skipping all write operations on spindown")

        # Clean up (always performed)
        self.vector_store.close()
        self.database.close()
        self.sandbox.cleanup()

        # Reset state
        self.config.agent_id = None
        self.config.session_id = None
        self.config.agent_name = None
        self._initialized = False

    def get_memory_note(self) -> str:
        """
        Get the current memory note for this agent.

        The memory note is a concise summary of what's stored in the agent's
        persistent memory. It's designed to be injected into the agent's
        system prompt to provide immediate context about available knowledge.

        Returns:
            The memory note string, or a default message if not available.

        Example:
            ```python
            from memlearn import MemFS
            from memlearn.prompts import get_memfs_system_prompt_with_note

            with MemFS.for_agent("my-agent") as memfs:
                memory_note = memfs.get_memory_note()
                system_prompt = f'''You are a helpful assistant.

                {get_memfs_system_prompt_with_note(memory_note)}
                '''
                # Use system_prompt with your LLM...
            ```
        """
        if not self.config.agent_id:
            return "This memory is empty. No files or insights have been stored yet."

        agent = self.database.get_agent_by_id(self.config.agent_id)
        if agent is None:
            return "This memory is empty. No files or insights have been stored yet."

        return agent.memory_note

    def _generate_memory_note(self) -> str | None:
        """
        Generate an updated memory note using the LLM.

        This method:
        1. Gets the current memory note
        2. Lists the entire filesystem structure
        3. Uses the LLM to generate a concise, updated note

        Returns:
            The new memory note, or None if generation failed or was skipped.
        """
        self._debug_log("_generate_memory_note() called")
        if self.llm is None:
            self._debug_log("_generate_memory_note() returning None: llm is None")
            return None

        if not self.config.agent_id:
            self._debug_log("_generate_memory_note() returning None: agent_id is None")
            return None

        # Get current agent and memory note
        agent = self.database.get_agent_by_id(self.config.agent_id)
        if agent is None:
            self._debug_log(
                "_generate_memory_note() returning None: agent not found in database"
            )
            return None

        current_note = agent.memory_note
        self._debug_log(
            f"Current memory note: {current_note[:100]}..."
            if len(current_note) > 100
            else f"Current memory note: {current_note}"
        )

        # Generate filesystem listing
        fs_listing = self._generate_fs_listing_for_note()
        self._debug_log(
            f"Filesystem listing ({len(fs_listing)} chars):\n{fs_listing[:500]}..."
            if len(fs_listing) > 500
            else f"Filesystem listing:\n{fs_listing}"
        )

        # Get prompts
        max_tokens = self.config.llm.memory_note_max_tokens
        system_prompt, user_prompt = get_memory_note_prompts(
            current_note=current_note,
            fs_listing=fs_listing,
        )
        self._debug_log(f"Calling LLM with max_tokens={max_tokens}")

        try:
            response = self.llm.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            self._debug_log(
                f"LLM response received, content length: {len(response.content)}"
            )

            return response.content.strip()

        except Exception as e:
            self._debug_log(
                f"_generate_memory_note() exception: {type(e).__name__}: {e}"
            )
            # Don't fail spindown if note generation fails
            return None

    def _generate_fs_listing_for_note(self) -> str:
        """Generate a filesystem listing formatted for memory note generation."""
        lines = []

        def list_recursive(path: str, depth: int = 0) -> None:
            try:
                items = self.sandbox.list_directory(path)
            except Exception:
                return

            for item in sorted(items):
                # Always exclude .git directory (internal version control)
                if item == ".git":
                    continue
                item_path = f"{path}/{item}" if path != "/" else f"/{item}"
                is_dir = self.sandbox.is_dir(item_path)
                indent = "  " * depth

                # Get metadata for description
                meta = self.database.get_metadata(item_path)
                desc = ""
                if meta and meta.description:
                    desc = f" - {meta.description[:60]}{'...' if len(meta.description) > 60 else ''}"

                if is_dir:
                    lines.append(f"{indent}📁 {item}/{desc}")
                    # Recurse into directories (limit depth to avoid huge outputs)
                    if depth < 3:
                        list_recursive(item_path, depth + 1)
                else:
                    size = self.sandbox.get_size(item_path)
                    size_str = self._format_size(size)
                    lines.append(f"{indent}📄 {item} ({size_str}){desc}")

        list_recursive("/")

        if not lines:
            return "(Empty filesystem)"

        return "\n".join(lines)

    def _update_memory_note(self, new_note: str) -> None:
        """Update the agent's memory note in the database."""
        self._debug_log(
            f"_update_memory_note() called with note length: {len(new_note)}"
        )
        if not self.config.agent_id:
            self._debug_log("_update_memory_note() returning: agent_id is None")
            return

        agent = self.database.get_agent_by_id(self.config.agent_id)
        if agent is None:
            self._debug_log(
                "_update_memory_note() returning: agent not found in database"
            )
            return

        self._debug_log(
            f"Updating agent {agent.name} (id={agent.agent_id}) memory_note"
        )
        agent.memory_note = new_note
        self.database.update_agent(agent)
        self._debug_log("Memory note updated successfully in database")

    @classmethod
    def for_agent(
        cls,
        agent_name: str,
        config: MemLearnConfig | None = None,
        read_only: bool = False,
    ) -> "MemFS":
        """
        Create a MemFS instance for an agent and start a new session.

        This is the primary entry point for developers. It creates the agent
        if it doesn't exist, then starts a new session with persistence.

        Args:
            agent_name: Unique name for the agent (e.g., "code-editor")
            config: Optional configuration. Uses default_persistent() if not provided.
            read_only: If True, creates a read-only MemFS instance that:
                - Does not modify the underlying filesystem on spindown
                - Does not store conversation history
                - Only provides read-only tools (read, list, search, peek, find)
                - Bash tool (if enabled) instructs LLM not to modify files

        Returns:
            Initialized MemFS instance ready for use.

        Example:
            ```python
            # Context manager (recommended)
            with MemFS.for_agent("code-editor") as memfs:
                tools = OpenAIToolProvider(memfs)
                # ... run agent session ...
            # Automatically persists and cleans up

            # Read-only mode for retrieval
            with MemFS.for_agent("code-editor", read_only=True) as memfs:
                tools = memfs.get_tool_provider()  # Returns read-only tools
                # ... query agent memory without modifications ...

            # Manual lifecycle
            memfs = MemFS.for_agent("code-editor")
            try:
                # ... run agent session ...
            finally:
                memfs.spindown()
            ```
        """
        # Use persistent config by default
        if config is None:
            config = MemLearnConfig.default_persistent()

        memfs = cls(config=config, read_only=read_only)
        memfs.spinup(agent_name)
        return memfs

    def get_tool_provider(
        self,
        provider: str = "langchain",
        tool_prefix: str = "memfs",
        enable_bash: bool = False,
    ):
        """
        Get a tool provider configured appropriately for this MemFS instance.

        The returned provider automatically respects the read_only mode:
        - In read-only mode: only read/search tools are provided (read, list, search, peek, find)
        - In read-write mode: all tools are provided

        Args:
            provider: The tool provider type. Currently supports "langchain".
            tool_prefix: Prefix for tool names (e.g., "memfs_read").
            enable_bash: Whether to enable the bash tool.
                In read-only mode, bash tool description instructs LLM not to modify files.

        Returns:
            A configured tool provider instance.

        Example:
            ```python
            # Read-write mode
            with MemFS.for_agent("my-agent") as memfs:
                provider = memfs.get_tool_provider(enable_bash=True)
                tools = provider.get_tools()

            # Read-only mode (for retrieval/queries)
            with MemFS.for_agent("my-agent", read_only=True) as memfs:
                provider = memfs.get_tool_provider()  # Only read tools
                tools = provider.get_tools()
            ```
        """
        if provider == "langchain":
            from memlearn.tools.langchain_tools import LangChainToolProvider

            return LangChainToolProvider(
                memfs=self,
                tool_prefix=tool_prefix,
                enable_bash=enable_bash,
                read_only=self.read_only,
            )
        else:
            raise ValueError(
                f"Unknown tool provider: {provider}. Supported: 'langchain'"
            )

    # =========================================================================
    # Conversation History Methods
    # =========================================================================

    def _summarize_conversation(self) -> str | None:
        """
        Generate a summary of the current conversation using the configured LLM.

        Returns:
            The conversation summary, or None if summarization failed or was skipped.
        """
        if self.llm is None:
            return None

        # Get the conversation messages
        messages = self.get_conversation_history()
        if not messages:
            return None

        try:
            # Use the OpenAI LLM's specialized method if available
            if hasattr(self.llm, "summarize_conversation"):
                return self.llm.summarize_conversation(
                    messages=messages,
                    agent_name=self.config.agent_name,
                    session_context=f"Session ID: {self.config.session_id}",
                )

            # Fallback to generic summarize method
            # Build conversation text
            conversation_parts = []
            for msg in messages:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                conversation_parts.append(f"{role}: {content}")

            conversation_text = "\n\n".join(conversation_parts)

            return self.llm.summarize(
                content=conversation_text,
                context=f"Agent conversation for '{self.config.agent_name}'",
            )

        except Exception:
            # Don't fail spindown if summarization fails
            return None

    def _init_conversation_history(self) -> None:
        """Create CURRENT.md with session metadata. Called during spinup."""
        if not self.config.session_id or not self.config.agent_id:
            return

        history = ConversationHistory.create(
            session_id=self.config.session_id,
            agent_id=self.config.agent_id,
            agent_name=self.config.agent_name or "unknown",
        )

        # Write to file in markdown format (bypass permissions since this is system operation)
        content = history.to_markdown()
        self.sandbox.write_file("raw/conversation-history/CURRENT.md", content)

        # Create metadata for the file
        metadata = NodeMetadata(
            path=CONVERSATION_HISTORY_PATH,
            node_type=NodeType.FILE,
            permissions=Permissions(readable=True, writable=False, executable=False),
            owner="system",
            file_type=FileType.MARKDOWN,
            description=f"Conversation history for session {self.config.session_id}",
            size_bytes=len(content.encode()),
            line_count=content.count("\n") + 1,
            extra={"session_id": self.config.session_id},
        )
        self.database.save_metadata(metadata)

    def _archive_conversation_history(self, summary: str | None = None) -> None:
        """
        Archive CURRENT.md with optional LLM-generated summary.

        This method:
        1. Appends the summary to the conversation file (if provided)
        2. Renames CURRENT.md to a timestamped filename
        3. Updates metadata with the summary
        4. Embeds the conversation history for semantic search (if auto_embed enabled)

        Args:
            summary: Optional conversation summary to store in metadata.
        """
        current_path = "raw/conversation-history/CURRENT.md"

        if not self.sandbox.exists(current_path):
            return

        # Read the history to get the start time
        try:
            content = self.sandbox.read_file(current_path)
            history = ConversationHistory.from_markdown(content)
            started_at = history.started_at

            # Add summary to the history if provided (as a final section)
            if summary:
                # Append summary section to the markdown
                summary_section = f"\n---\n\n" f"## Session Summary\n\n" f"{summary}\n"
                content = content + summary_section
                self.sandbox.write_file(current_path, content)

            # Create timestamp filename (ISO format with safe characters)
            dt = datetime.datetime.fromtimestamp(started_at)
            timestamp_str = dt.strftime("%Y-%m-%dT%H-%M-%S")
            archive_filename = f"{timestamp_str}.md"
            archive_path = f"raw/conversation-history/{archive_filename}"

            # Rename the file
            self.sandbox.move(current_path, archive_path)

            # Update metadata with summary
            old_metadata = self.database.get_metadata(CONVERSATION_HISTORY_PATH)
            if old_metadata:
                self.database.delete_metadata(CONVERSATION_HISTORY_PATH)
                old_metadata.path = f"/raw/conversation-history/{archive_filename}"
                old_metadata.description = (
                    f"Archived conversation history from {timestamp_str}"
                )
                # Store summary in metadata extra field
                if summary:
                    old_metadata.extra["summary"] = summary
                    old_metadata.extra["summarized_at"] = time.time()
                    old_metadata.extra["summary_model"] = (
                        self.llm.model if self.llm else None
                    )

                # Embed the conversation history for semantic search
                if self.config.auto_embed:
                    archive_content = self.sandbox.read_file(archive_path)
                    self._embed_file(old_metadata, archive_content)

                self.database.save_metadata(old_metadata)

        except Exception:
            # If anything fails, just leave the file as is
            pass

    def append_conversation_message(self, message: dict[str, Any]) -> None:
        """
        Append a message to the current session's conversation history.

        Call this after each LLM interaction to keep the history updated.
        In read-only mode, this is a no-op.

        Args:
            message: A message dict with role, content, and optional fields:
                - role: "user", "assistant", "system", "tool_call", "tool_result"
                - content: The message content
                - timestamp: Optional Unix timestamp (added automatically if missing)
                - tool_name: Optional tool name for tool_call/tool_result roles
        """
        # No-op in read-only mode
        if self.read_only:
            return

        current_path = "raw/conversation-history/CURRENT.md"

        if not self.sandbox.exists(current_path):
            return

        try:
            content = self.sandbox.read_file(current_path)
            history = ConversationHistory.from_markdown(content)

            # Add timestamp if not present
            if "timestamp" not in message:
                message = {**message, "timestamp": time.time()}

            history.append_message(message)

            new_content = history.to_markdown()
            self.sandbox.write_file(current_path, new_content)

            # Update metadata size
            metadata = self.database.get_metadata(CONVERSATION_HISTORY_PATH)
            if metadata:
                metadata.size_bytes = len(new_content.encode())
                metadata.line_count = new_content.count("\n") + 1
                metadata.timestamps.touch_modify()
                self.database.save_metadata(metadata)

        except Exception:
            pass

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """
        Read current session's full conversation history.

        Returns:
            List of message dicts from the conversation.
        """
        current_path = "raw/conversation-history/CURRENT.md"

        if not self.sandbox.exists(current_path):
            return []

        try:
            content = self.sandbox.read_file(current_path)
            history = ConversationHistory.from_markdown(content)
            return history.messages
        except Exception:
            return []

    def compact_conversation(
        self, summary: str, preserve_last_n: int = 10
    ) -> ToolResult:
        """
        Compact conversation history while preserving full history in file.

        This adds a compaction marker to the history file and returns
        compacted messages for use in the context window.

        Args:
            summary: A comprehensive summary of the conversation so far
            preserve_last_n: Number of recent messages to keep uncompacted

        Returns:
            ToolResult with compacted messages for the context window
        """
        current_path = "raw/conversation-history/CURRENT.md"

        if not self.sandbox.exists(current_path):
            return ToolResult(
                status="error",
                message="No active conversation history found.",
            )

        try:
            content = self.sandbox.read_file(current_path)
            history = ConversationHistory.from_markdown(content)

            if len(history.messages) <= preserve_last_n:
                return ToolResult(
                    status="success",
                    message="Conversation too short to compact.",
                    data={"messages": history.messages, "compacted": False},
                )

            # Add compaction marker
            history.add_compaction_marker(summary)

            # Save updated history with marker
            new_content = history.to_markdown()
            self.sandbox.write_file(current_path, new_content)

            # Build compacted messages for return
            # Start with a system message containing the summary
            compacted_messages = [
                {
                    "role": "system",
                    "content": f"[Previous conversation summary]\n{summary}\n[End of summary - recent messages follow]",
                }
            ]

            # Add the preserved recent messages
            recent_messages = history.messages[-preserve_last_n:]
            compacted_messages.extend(recent_messages)

            return ToolResult(
                status="success",
                message=f"Compacted conversation. Preserved {len(recent_messages)} recent messages.",
                data={
                    "messages": compacted_messages,
                    "compacted": True,
                    "original_count": len(history.messages),
                    "preserved_count": len(recent_messages),
                },
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Failed to compact conversation: {str(e)}",
            )

    # =========================================================================
    # Mount Operations
    # =========================================================================

    def mount_agent_memory(
        self, source_agent_name: str, mount_name: str | None = None
    ) -> ToolResult:
        """
        Mount another agent's memory folder for read access.

        Args:
            source_agent_name: Name of the agent whose memory to mount
            mount_name: Optional name for the mount point (defaults to agent name)

        Returns:
            ToolResult indicating success or failure
        """
        if not self.config.agent_id:
            return ToolResult(
                status="error",
                message="No active session. Use spinup() first.",
            )

        # Find the source agent
        source_agent = self.database.get_agent_by_name(source_agent_name)
        if source_agent is None:
            return ToolResult(
                status="error",
                message=f"Agent not found: {source_agent_name}",
            )

        if source_agent.agent_id == self.config.agent_id:
            return ToolResult(
                status="error",
                message="Cannot mount your own memory. Use /memory directly.",
            )

        # Determine mount path
        mount_name = mount_name or source_agent_name
        mount_path = f"/mnt/agents/{mount_name}"

        # Check if already mounted
        existing_mounts = self.database.get_mounts_for_agent(self.config.agent_id)
        for mount in existing_mounts:
            if mount.mount_path == mount_path:
                return ToolResult(
                    status="error",
                    message=f"Mount point already in use: {mount_path}",
                )

        # Get the source agent's persistent path
        source_persistent_path = self.config.sandbox.get_agent_persistent_path(
            source_agent.agent_id
        )
        source_memory_path = source_persistent_path / "memory"

        if not source_memory_path.exists():
            return ToolResult(
                status="error",
                message=f"Source agent has no persistent memory: {source_agent_name}",
            )

        # Create the mount point in sandbox
        sandbox_mount_path = f"mnt/agents/{mount_name}"
        self.sandbox.create_directory(sandbox_mount_path)

        # Copy the source memory into the mount point
        import shutil

        dest_path = Path(self.sandbox.root_path) / sandbox_mount_path
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(source_memory_path, dest_path)

        # Create mount record in database
        mount = MountInfo.create(
            agent_id=self.config.agent_id,
            mount_path=mount_path,
            source_type=MountSourceType.AGENT,
            source_ref=source_agent.agent_id,
        )
        self.database.create_mount(mount)

        # Create metadata for mount point
        metadata = NodeMetadata(
            path=mount_path,
            node_type=NodeType.FOLDER,
            permissions=Permissions(readable=True, writable=True, executable=False),
            owner="system",
            description=f"Mounted memory from agent: {source_agent_name}",
        )
        self.database.save_metadata(metadata)

        return ToolResult(
            status="success",
            message=f"Mounted {source_agent_name}'s memory at {mount_path}",
            data={"mount_path": mount_path, "source_agent": source_agent_name},
        )

    def unmount(self, mount_path: str) -> ToolResult:
        """
        Unmount a previously mounted folder.

        Args:
            mount_path: Path of the mount point (e.g., /mnt/agents/other-agent)

        Returns:
            ToolResult indicating success or failure
        """
        mount_path = self._normalize_path(mount_path)

        if not self.config.agent_id:
            return ToolResult(
                status="error",
                message="No active session.",
            )

        # Find the mount
        mounts = self.database.get_mounts_for_agent(self.config.agent_id)
        mount_to_remove = None
        for mount in mounts:
            if mount.mount_path == mount_path:
                mount_to_remove = mount
                break

        if mount_to_remove is None:
            return ToolResult(
                status="error",
                message=f"Mount not found: {mount_path}",
            )

        # Remove from sandbox
        sandbox_path = mount_path.lstrip("/")
        if self.sandbox.exists(sandbox_path):
            self.sandbox.delete_directory(sandbox_path, recursive=True)

        # Remove mount record
        self.database.delete_mount(mount_to_remove.mount_id)

        # Remove metadata
        self.database.delete_metadata(mount_path)

        return ToolResult(
            status="success",
            message=f"Unmounted: {mount_path}",
        )

    def _remount_saved_mounts(self) -> None:
        """
        Remount previously mounted folders. Called during spinup.
        """
        if not self.config.agent_id:
            return

        mounts = self.database.get_mounts_for_agent(self.config.agent_id)

        for mount in mounts:
            if mount.source_type == MountSourceType.AGENT:
                # Get source agent's persistent memory
                source_persistent_path = self.config.sandbox.get_agent_persistent_path(
                    mount.source_ref
                )
                source_memory_path = source_persistent_path / "memory"

                if source_memory_path.exists():
                    # Create mount point and copy
                    sandbox_mount_path = mount.mount_path.lstrip("/")
                    self.sandbox.create_directory(sandbox_mount_path)

                    import shutil

                    dest_path = Path(self.sandbox.root_path) / sandbox_mount_path
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_memory_path, dest_path)

    def list_mounts(self) -> ToolResult:
        """
        List all current mounts for this agent.

        Returns:
            ToolResult with list of mounts
        """
        if not self.config.agent_id:
            return ToolResult(
                status="error",
                message="No active session.",
            )

        mounts = self.database.get_mounts_for_agent(self.config.agent_id)

        mount_list = []
        for mount in mounts:
            mount_info = {
                "mount_path": mount.mount_path,
                "source_type": mount.source_type.value,
                "source_ref": mount.source_ref,
            }

            # Try to get source name for agent mounts
            if mount.source_type == MountSourceType.AGENT:
                source_agent = self.database.get_agent_by_id(mount.source_ref)
                if source_agent:
                    mount_info["source_name"] = source_agent.name

            mount_list.append(mount_info)

        return ToolResult(
            status="success",
            message=f"Found {len(mount_list)} mount(s)",
            data={"mounts": mount_list},
        )

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
                owner=self.config.agent_id or "system",
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

            # Git commit
            self._git_commit_all(f"create: {path}")

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
            self._log_operation(
                "edit_file",
                path,
                {
                    "old_length": len(old_string),
                    "new_length": len(new_string),
                },
            )
            self._create_version_snapshot(f"Edited file: {path}", [path])

            # Git commit
            self._git_commit_all(f"edit: {path}")

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

            # Git commit
            self._git_commit_all(f"delete: {path}")

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
                owner=self.config.agent_id or "system",
                description=description,
                tags=tags or [],
            )

            # Embed description if provided
            if self.config.auto_embed and description:
                metadata.embedding = self.embedder.embed(description)
                self._add_to_vector_store(
                    path,
                    metadata.embedding,
                    {
                        "path": path,
                        "type": "folder",
                        "description": description,
                    },
                )

            self.database.save_metadata(metadata)

            # Log and version
            self._log_operation("create_directory", path)
            self._create_version_snapshot(f"Created directory: {path}", [path])

            # Git commit
            self._git_commit_all(f"mkdir: {path}")

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
            # Always exclude .git directory (internal version control)
            if item == ".git":
                continue
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

            # Git commit
            self._git_commit_all(f"move: {src} -> {dst}")

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

            results.append(
                SearchResult(
                    path=vr.metadata.get("path", ""),
                    score=vr.score,
                    content_snippet=vr.metadata.get("content", ""),
                    chunk_index=vr.metadata.get("chunk_index"),
                    start_line=vr.metadata.get("start_line"),
                    end_line=vr.metadata.get("end_line"),
                )
            )

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

                    results.append(
                        SearchResult(
                            path=file_path,
                            score=min(score, 1.0),
                            content_snippet=snippet,
                            start_line=start + 1,
                            end_line=end,
                        )
                    )

            except Exception:
                pass

        def search_recursive(dir_path: str) -> None:
            try:
                items = self.sandbox.list_directory(dir_path)
                for item in items:
                    # Always exclude .git directory (internal version control)
                    if item == ".git":
                        continue
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

        documents = [f"{r.path}: {r.content_snippet or ''}" for r in results]

        try:
            reranked = self.reranker.rerank(query, documents, top_n=top_k)

            reranked_results = []
            for rr in reranked:
                original = results[rr.index]
                reranked_results.append(
                    SearchResult(
                        path=original.path,
                        score=rr.score,
                        content_snippet=original.content_snippet,
                        chunk_index=original.chunk_index,
                        start_line=original.start_line,
                        end_line=original.end_line,
                    )
                )

            return reranked_results

        except Exception:
            # Fall back to original ordering
            return results[:top_k]

    # =========================================================================
    # Peek Operation
    # =========================================================================

    def find(
        self,
        pattern: str,
        path: str = "/",
        match_type: str = "glob",
        include_dirs: bool = False,
        max_results: int = 50,
        offset: int = 0,
        sort_by: str = "path",
        sort_order: str = "asc",
    ) -> ToolResult:
        """
        Find files by name pattern with LLM-friendly pagination and metadata.

        This tool finds files matching a name pattern and returns metadata in a
        format optimized for LLM context windows. It supports pagination to avoid
        overwhelming the context with too many results.

        Args:
            pattern: The pattern to match against file names.
                - For glob: supports *, ?, ** (e.g., "*.md", "test_*.py", "**/*.json")
                - For regex: full regex pattern (e.g., ".*\\.md$", "test_\\d+\\.py")
                - For exact: exact filename match (e.g., "README.md")
            path: Base path to search from. Defaults to "/" (entire MemFS).
            match_type: Pattern matching type - "glob", "regex", or "exact".
            include_dirs: If True, also include matching directories.
            max_results: Maximum number of results to return (for pagination).
                Default is 50. Set to -1 for unlimited (use with caution).
            offset: Number of results to skip (for pagination). Default is 0.
            sort_by: Sort results by "path", "name", "size", "modified", or "created".
            sort_order: Sort order - "asc" (ascending) or "desc" (descending).

        Returns:
            ToolResult with:
            - matches: List of matching files with metadata
            - total_count: Total number of matches (before pagination)
            - returned_count: Number of results returned in this response
            - has_more: Whether there are more results after this page
            - next_offset: Offset to use to get the next page
            - formatted: Human-readable summary of results
        """
        path = self._normalize_path(path)

        if not self.sandbox.exists(path):
            return ToolResult(
                status="error",
                message=f"Path not found: {path}",
            )

        if not self.sandbox.is_dir(path):
            return ToolResult(
                status="error",
                message=f"Path is not a directory: {path}. Use a directory path to search.",
            )

        try:
            # Compile pattern based on match type
            if match_type == "regex":
                try:
                    compiled_pattern = re.compile(pattern)
                except re.error as e:
                    return ToolResult(
                        status="error",
                        message=f"Invalid regex pattern: {e}",
                    )
            elif match_type == "glob":
                # Convert glob to regex
                regex_pattern = self._glob_to_regex(pattern)
                compiled_pattern = re.compile(regex_pattern)
            else:  # exact
                compiled_pattern = None  # Use direct comparison

            # Collect all matching files
            all_matches = self._find_files_recursive(
                path, pattern, compiled_pattern, match_type, include_dirs
            )

            total_count = len(all_matches)

            # Sort results
            all_matches = self._sort_find_results(all_matches, sort_by, sort_order)

            # Apply pagination
            if max_results == -1:
                paginated_matches = all_matches[offset:]
            else:
                paginated_matches = all_matches[offset : offset + max_results]

            returned_count = len(paginated_matches)
            has_more = offset + returned_count < total_count
            next_offset = offset + returned_count if has_more else None

            # Format results with metadata
            results = []
            for match in paginated_matches:
                file_path = match["path"]
                metadata = self.database.get_metadata(file_path)

                result_entry = {
                    "path": file_path,
                    "name": match["name"],
                    "type": "folder" if match["is_dir"] else "file",
                    "size_bytes": match.get("size", 0),
                    "size_human": self._format_size(match.get("size", 0)),
                }

                if metadata:
                    result_entry.update(
                        {
                            "description": metadata.description or None,
                            "tags": metadata.tags if metadata.tags else None,
                            "file_type": (
                                metadata.file_type.value
                                if not match["is_dir"]
                                else None
                            ),
                            "line_count": (
                                metadata.line_count if not match["is_dir"] else None
                            ),
                            "created": self._format_timestamp(
                                metadata.timestamps.created_at
                            ),
                            "modified": self._format_timestamp(
                                metadata.timestamps.modified_at
                            ),
                            "permissions": f"{'r' if metadata.permissions.readable else '-'}"
                            f"{'w' if metadata.permissions.writable else '-'}"
                            f"{'x' if metadata.permissions.executable else '-'}",
                        }
                    )
                else:
                    result_entry.update(
                        {
                            "description": None,
                            "tags": None,
                            "file_type": (
                                self._detect_file_type(file_path, "").value
                                if not match["is_dir"]
                                else None
                            ),
                            "line_count": None,
                            "created": None,
                            "modified": None,
                            "permissions": None,
                        }
                    )

                # Remove None values to keep output clean
                result_entry = {k: v for k, v in result_entry.items() if v is not None}
                results.append(result_entry)

            # Build formatted output for LLM readability
            output_lines = self._format_find_output(
                pattern,
                match_type,
                path,
                results,
                total_count,
                returned_count,
                offset,
                has_more,
                next_offset,
            )

            # Log operation
            self._log_operation(
                "find",
                path,
                {
                    "pattern": pattern,
                    "match_type": match_type,
                    "total_matches": total_count,
                    "returned": returned_count,
                },
            )

            return ToolResult(
                status="success",
                message=f"Found {total_count} match{'es' if total_count != 1 else ''} for pattern '{pattern}'",
                data={
                    "matches": results,
                    "total_count": total_count,
                    "returned_count": returned_count,
                    "offset": offset,
                    "has_more": has_more,
                    "next_offset": next_offset,
                    "pattern": pattern,
                    "match_type": match_type,
                    "search_path": path,
                    "formatted": "\n".join(output_lines),
                },
            )

        except Exception as e:
            return ToolResult(
                status="error",
                message=f"Find operation failed: {str(e)}",
            )

    def _glob_to_regex(self, pattern: str) -> str:
        """
        Convert a glob pattern to a regex pattern.

        Supports:
        - * : matches any characters except /
        - ** : matches any characters including /
        - ? : matches any single character except /
        - [seq] : matches any character in seq
        - [!seq] : matches any character not in seq
        """
        i = 0
        n = len(pattern)
        regex_parts = []

        while i < n:
            c = pattern[i]
            if c == "*":
                if i + 1 < n and pattern[i + 1] == "*":
                    # ** matches anything including path separators
                    regex_parts.append(".*")
                    i += 2
                    # Skip following / if present
                    if i < n and pattern[i] == "/":
                        i += 1
                else:
                    # * matches anything except /
                    regex_parts.append("[^/]*")
                    i += 1
            elif c == "?":
                regex_parts.append("[^/]")
                i += 1
            elif c == "[":
                # Find the closing bracket
                j = i + 1
                if j < n and pattern[j] == "!":
                    j += 1
                if j < n and pattern[j] == "]":
                    j += 1
                while j < n and pattern[j] != "]":
                    j += 1
                if j >= n:
                    regex_parts.append("\\[")
                else:
                    stuff = pattern[i + 1 : j]
                    if stuff[0:1] == "!":
                        stuff = "^" + stuff[1:]
                    elif stuff[0:1] == "^":
                        stuff = "\\" + stuff
                    regex_parts.append(f"[{stuff}]")
                    j += 1
                i = j
            elif c in ".^$+{}|()":
                regex_parts.append("\\" + c)
                i += 1
            else:
                regex_parts.append(c)
                i += 1

        return "".join(regex_parts) + "$"

    def _find_files_recursive(
        self,
        dir_path: str,
        pattern: str,
        compiled_pattern: re.Pattern | None,
        match_type: str,
        include_dirs: bool,
    ) -> list[dict]:
        """Recursively find files matching the pattern."""
        matches = []

        try:
            items = self.sandbox.list_directory(dir_path)
        except Exception:
            return matches

        for item in items:
            # Always exclude .git directory (internal version control)
            if item == ".git":
                continue
            item_path = f"{dir_path}/{item}" if dir_path != "/" else f"/{item}"
            is_dir = self.sandbox.is_dir(item_path)

            # Check if item matches pattern
            is_match = False
            if match_type == "exact":
                is_match = item == pattern
            elif compiled_pattern:
                # For glob/regex, match against just the filename
                is_match = bool(compiled_pattern.match(item))
                # Also try matching against full path for ** patterns
                if not is_match and "**" in pattern:
                    is_match = bool(compiled_pattern.match(item_path.lstrip("/")))

            # Add to matches if appropriate
            if is_match and (include_dirs or not is_dir):
                entry = {
                    "path": item_path,
                    "name": item,
                    "is_dir": is_dir,
                }
                if not is_dir:
                    entry["size"] = self.sandbox.get_size(item_path)

                    # Get modification time from metadata if available
                    meta = self.database.get_metadata(item_path)
                    if meta:
                        entry["modified_at"] = meta.timestamps.modified_at
                        entry["created_at"] = meta.timestamps.created_at
                matches.append(entry)

            # Recurse into directories
            if is_dir:
                child_matches = self._find_files_recursive(
                    item_path, pattern, compiled_pattern, match_type, include_dirs
                )
                matches.extend(child_matches)

        return matches

    def _sort_find_results(
        self, matches: list[dict], sort_by: str, sort_order: str
    ) -> list[dict]:
        """Sort find results by the specified field."""
        reverse = sort_order.lower() == "desc"

        if sort_by == "name":
            return sorted(matches, key=lambda x: x["name"].lower(), reverse=reverse)
        elif sort_by == "size":
            return sorted(matches, key=lambda x: x.get("size", 0), reverse=reverse)
        elif sort_by == "modified":
            return sorted(
                matches, key=lambda x: x.get("modified_at", 0), reverse=reverse
            )
        elif sort_by == "created":
            return sorted(
                matches, key=lambda x: x.get("created_at", 0), reverse=reverse
            )
        else:  # default: path
            return sorted(matches, key=lambda x: x["path"].lower(), reverse=reverse)

    def _format_find_output(
        self,
        pattern: str,
        match_type: str,
        search_path: str,
        results: list[dict],
        total_count: int,
        returned_count: int,
        offset: int,
        has_more: bool,
        next_offset: int | None,
    ) -> list[str]:
        """Format find results for LLM-friendly output."""
        lines = []

        # Header with summary
        lines.append(f"=== Find Results ===")
        lines.append(f"Pattern: '{pattern}' ({match_type})")
        lines.append(f"Search path: {search_path}")
        lines.append(f"Total matches: {total_count}")

        if total_count > returned_count:
            lines.append(
                f"Showing: {offset + 1}-{offset + returned_count} of {total_count}"
            )
        lines.append("")

        if not results:
            lines.append("No matches found.")
            return lines

        # Results table
        lines.append("Matches:")
        lines.append("-" * 60)

        for i, result in enumerate(results, start=offset + 1):
            type_indicator = "📁" if result.get("type") == "folder" else "📄"
            path = result["path"]
            size = result.get("size_human", "")

            # Main line with path and size
            lines.append(f"{i}. {type_indicator} {path}")

            # Metadata on indented lines (only if present)
            meta_parts = []
            if size:
                meta_parts.append(f"size: {size}")
            if result.get("line_count"):
                meta_parts.append(f"lines: {result['line_count']}")
            if result.get("file_type"):
                meta_parts.append(f"type: {result['file_type']}")
            if result.get("modified"):
                meta_parts.append(f"modified: {result['modified']}")

            if meta_parts:
                lines.append(f"   {' | '.join(meta_parts)}")

            if result.get("description"):
                desc = result["description"]
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                lines.append(f"   desc: {desc}")

            if result.get("tags"):
                lines.append(f"   tags: {', '.join(result['tags'])}")

        # Pagination info
        if has_more:
            lines.append("-" * 60)
            lines.append(
                f"More results available. Use offset={next_offset} to see next page."
            )

        return lines

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
        output_lines.append(
            f"  Tags: {', '.join(metadata.tags) if metadata.tags else '(none)'}"
        )

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
    # Git Version Control Methods
    # =========================================================================

    def _git_enabled(self) -> bool:
        """Check if git version control is enabled and we're not in read-only mode."""
        return self.config.sandbox.version_control_enabled and not self.read_only

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the sandbox root directory."""
        cmd = ["git"] + list(args)
        return subprocess.run(
            cmd,
            cwd=self.sandbox.root_path,
            capture_output=True,
            text=True,
            check=check,
        )

    def _git_init(self) -> None:
        """Initialize git repository in the sandbox if version control is enabled."""
        if not self._git_enabled():
            return

        git_dir = Path(self.sandbox.root_path) / ".git"
        if git_dir.exists():
            return

        try:
            self._run_git("init", "-b", "main")
            self._run_git("config", "user.email", "memfs@memlearn.local")
            self._run_git("config", "user.name", "MemFS")
            self._debug_log("Git repository initialized")
        except subprocess.CalledProcessError as e:
            self._debug_log(f"Failed to initialize git: {e.stderr}")

    def _git_commit_all(self, message: str) -> bool:
        """Stage all changes and commit with the given message."""
        if not self._git_enabled():
            return False

        try:
            self._run_git("add", "-A")
            result = self._run_git("status", "--porcelain", check=False)
            if not result.stdout.strip():
                return False
            self._run_git("commit", "-m", message, "--allow-empty-message")
            self._debug_log(f"Git commit: {message[:50]}...")
            return True
        except subprocess.CalledProcessError as e:
            self._debug_log(f"Git commit failed: {e.stderr}")
            return False

    def _git_create_session_branch(self) -> None:
        """Create a new branch for the current session off main."""
        if not self._git_enabled():
            return

        try:
            dt = datetime.datetime.now()
            branch_name = f"session-{dt.strftime('%Y%m%d-%H%M%S')}"
            self._session_branch_name = branch_name
            self._run_git("checkout", "-b", branch_name)
            self._debug_log(f"Created session branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            self._debug_log(f"Failed to create session branch: {e.stderr}")

    def _git_session_end_commit(self, summary: str | None = None) -> None:
        """Commit at the end of a session with datetime and optional summary."""
        if not self._git_enabled():
            return

        dt = datetime.datetime.now()
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")

        if summary:
            message = f"Session end: {timestamp}\n\n{summary}"
        else:
            message = f"Session end: {timestamp}"

        self._git_commit_all(message)

    def _git_get_session_commit_count(self) -> int:
        """Get the number of commits in the current session (since branching from main)."""
        if not self._git_enabled():
            return 0

        try:
            result = self._run_git("rev-list", "--count", "main..HEAD", check=False)
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
        except (subprocess.CalledProcessError, ValueError):
            return 0

    def undo(self, count: int = 1) -> ToolResult:
        """
        Undo recent filesystem changes within the current session.

        Uses git to revert changes made during this session. Cannot undo
        changes from previous sessions or before session start.

        Args:
            count: Number of changes to undo. Use -1 to undo all changes
                in the current session. If count exceeds the number of
                available changes, reverts all available changes.

        Returns:
            ToolResult indicating success or failure with details.
        """
        if not self._git_enabled():
            return ToolResult(
                status="error",
                message="Version control is disabled. Cannot undo changes.",
            )

        try:
            available_commits = self._git_get_session_commit_count()

            if available_commits == 0:
                return ToolResult(
                    status="success",
                    message="No changes to undo in this session.",
                    data={"undone_count": 0, "remaining_changes": 0},
                )

            if count == -1 or count > available_commits:
                count = available_commits

            self._run_git("reset", "--hard", f"HEAD~{count}")

            remaining = self._git_get_session_commit_count()

            self._debug_log(f"Undone {count} changes, {remaining} remaining")

            return ToolResult(
                status="success",
                message=f"Successfully undone {count} change(s).",
                data={
                    "undone_count": count,
                    "remaining_changes": remaining,
                },
            )

        except subprocess.CalledProcessError as e:
            return ToolResult(
                status="error",
                message=f"Failed to undo changes: {e.stderr}",
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

        # Check if this is a conversation history file
        is_conversation_history = metadata.path.startswith(
            "/raw/conversation-history/"
        ) and metadata.path.endswith(".md")

        if is_conversation_history:
            # Use specialized conversation chunking
            chunks = chunk_conversation_history(
                content,
                chunk_size=self.config.embedder.chunk_size
                * 3,  # Larger chunks for conversations
                chunk_overlap=self.config.embedder.chunk_overlap
                * 4,  # More overlap for context
                min_chunk_size=self.config.embedder.chunk_size // 2,
            )

            metadata.chunks = []
            chunk_texts = []
            chunk_metadata_list = []

            for i, (chunk_text, start_line, end_line, conv_metadata) in enumerate(
                chunks
            ):
                # Create optimized text for embedding
                embed_text = create_chunk_for_embedding(chunk_text, conv_metadata)
                chunk = Chunk.from_content(chunk_text, start_line, end_line)
                metadata.chunks.append(chunk)
                chunk_texts.append(embed_text)

                # Rich metadata for conversation chunks
                chunk_meta = {
                    "path": metadata.path,
                    "chunk_index": i,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": chunk_text[:200],
                    "content_type": "conversation_history",
                    "datetime_start": conv_metadata.get("datetime_start", "unknown"),
                    "datetime_end": conv_metadata.get("datetime_end", "unknown"),
                    "roles": ",".join(conv_metadata.get("roles", [])),
                    "is_partial_turn": conv_metadata.get("is_partial_turn", False),
                    "turn_count": conv_metadata.get("turn_count", 0),
                }
                chunk_metadata_list.append(chunk_meta)
        else:
            # Use standard chunking for other files
            chunks = self.embedder.chunk_text(
                content,
                chunk_size=self.config.embedder.chunk_size,
                overlap=self.config.embedder.chunk_overlap,
            )

            metadata.chunks = []
            chunk_texts = []
            chunk_metadata_list = []

            for i, (chunk_text, start_line, end_line) in enumerate(chunks):
                chunk = Chunk.from_content(chunk_text, start_line, end_line)
                metadata.chunks.append(chunk)
                chunk_texts.append(chunk_text)
                chunk_metadata_list.append(
                    {
                        "path": metadata.path,
                        "chunk_index": i,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": chunk_text[:200],  # Store snippet for search results
                    }
                )

        # Batch embed chunks
        if chunk_texts:
            embeddings = self.embedder.embed_batch(chunk_texts)
            for i, embedding in enumerate(embeddings):
                metadata.chunks[i].embedding = embedding

            # Add to vector store
            ids = [f"{metadata.path}:chunk:{i}" for i in range(len(metadata.chunks))]
            self.vector_store.add(ids, embeddings, chunk_metadata_list, chunk_texts)

        # Add description embedding to vector store
        if metadata.embedding:
            self.vector_store.add(
                [f"{metadata.path}:description"],
                [metadata.embedding],
                [
                    {
                        "path": metadata.path,
                        "type": "description",
                        "content": metadata.description,
                    }
                ],
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
            agent_id=self.config.agent_id or "system",
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
            agent_id=self.config.agent_id or "system",
        )
        self.database.save_snapshot(snapshot)


# =========================================================================
# Factory Functions
# =========================================================================


def create_memfs(config: MemLearnConfig | None = None) -> MemFS:
    """
    Create and initialize a new MemFS instance (ephemeral, no persistence).

    For persistent sessions, use MemFS.for_agent() instead.

    Args:
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        Initialized MemFS instance.
    """
    memfs = MemFS(config=config)
    memfs.initialize()
    return memfs


def load_agent(
    agent_name: str,
    config: MemLearnConfig | None = None,
) -> MemFS:
    """
    Load a MemFS instance for an agent with persistence.

    This is an alias for MemFS.for_agent().

    Args:
        agent_name: Unique name for the agent (e.g., "code-editor")
        config: Optional configuration. Uses default_persistent() if not provided.

    Returns:
        MemFS instance with persistence enabled.
    """
    return MemFS.for_agent(agent_name, config)


# Keep backward compatibility
def load_memfs(
    agent_id: str | None = None,
    config: MemLearnConfig | None = None,
) -> MemFS:
    """
    Load a MemFS instance (deprecated - use load_agent() for persistence).

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
