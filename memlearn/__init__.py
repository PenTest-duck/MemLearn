"""
MemLearn - An open-source agentic memory and continual learning system.

MemLearn provides a filesystem-like architecture (MemFS) for LLM agents to
store, manage, and retrieve memory with semantic search capabilities.

Basic Usage:
    from memlearn import create_memfs, get_openai_tools, execute_openai_tool

    # Create and initialize MemFS
    memfs = create_memfs()

    # Get tools for your agent
    tools = get_openai_tools(memfs)

    # Execute tool calls from your agent
    result = execute_openai_tool(memfs, "memfs_read", {"path": "/memory/notes.md"})

    # Clean up when done
    memfs.close()

Context Manager Usage:
    from memlearn import MemFS

    with MemFS() as memfs:
        # Use memfs...
        memfs.create_file("/memory/test.md", "Hello, World!")
"""

from memlearn.config import MemLearnConfig
from memlearn.memfs import MemFS, create_memfs, load_memfs
from memlearn.prompts import (
    MEMFS_SYSTEM_PROMPT,
    get_custom_memfs_prompt,
    get_memfs_system_prompt,
)
from memlearn.tools import execute_openai_tool, get_openai_tools
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

__version__ = "0.0.1"

__all__ = [
    # Core classes
    "MemFS",
    "MemLearnConfig",
    # Factory functions
    "create_memfs",
    "load_memfs",
    # Tool helpers
    "get_openai_tools",
    "execute_openai_tool",
    # Prompts
    "MEMFS_SYSTEM_PROMPT",
    "get_memfs_system_prompt",
    "get_custom_memfs_prompt",
    # Types
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
