"""
MemLearn - An open-source agentic memory and continual learning system.

MemLearn provides a filesystem-like architecture (MemFS) for LLM agents to
store, manage, and retrieve memory with semantic search capabilities.

Persistent Agent Usage (Recommended):
    from memlearn import MemFS
    from memlearn.tools import OpenAIToolProvider

    # Use context manager for automatic persistence
    with MemFS.for_agent("code-editor") as memfs:
        tools = OpenAIToolProvider(memfs)
        # ... run agent session ...
    # Automatically persists and cleans up

Ephemeral Usage (No persistence):
    from memlearn import create_memfs, get_openai_tools, execute_openai_tool

    # Create ephemeral MemFS (for testing/development)
    memfs = create_memfs()

    # Get tools for your agent
    tools = get_openai_tools(memfs)

    # Execute tool calls from your agent
    result = execute_openai_tool(memfs, "memfs_read", {"path": "/memory/notes.md"})

    # Clean up when done
    memfs.close()
"""

from memlearn.config import MemLearnConfig
from memlearn.memfs import MemFS, create_memfs, load_agent
from memlearn.prompts import (
    MEMFS_SYSTEM_PROMPT,
    get_custom_memfs_prompt,
    get_memfs_system_prompt,
    get_memfs_system_prompt_with_note,
)
from memlearn.tools import execute_openai_tool, get_openai_tools
from memlearn.types import (
    Agent,
    Chunk,
    CompactionMarker,
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

__version__ = "0.0.1"

__all__ = [
    # Core classes
    "MemFS",
    "MemLearnConfig",
    # Factory functions
    "create_memfs",
    "load_agent",
    # Tool helpers
    "get_openai_tools",
    "execute_openai_tool",
    # Prompts
    "MEMFS_SYSTEM_PROMPT",
    "get_memfs_system_prompt",
    "get_memfs_system_prompt_with_note",
    "get_custom_memfs_prompt",
    # Entity types
    "Agent",
    "Session",
    "SessionStatus",
    "MountInfo",
    "MountSourceType",
    "ConversationHistory",
    "CompactionMarker",
    # MemFS types
    "NodeMetadata",
    "NodeType",
    "FileType",
    "Permissions",
    "Timestamps",
    "Chunk",
    "MemFSLog",
    "SearchResult",
    "ToolResult",
    "VersionSnapshot",
]
