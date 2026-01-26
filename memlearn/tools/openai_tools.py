"""
OpenAI-compatible tool definitions for MemFS operations.
"""

from __future__ import annotations

import json
from typing import Any

from memlearn.memfs import MemFS
from memlearn.tools.base import BaseToolProvider


class OpenAIToolProvider(BaseToolProvider):
    """OpenAI function calling compatible tool provider for MemFS."""

    def __init__(self, memfs: MemFS, tool_prefix: str = "memfs"):
        """
        Initialize the OpenAI tool provider.

        Args:
            memfs: The MemFS instance to operate on.
            tool_prefix: Prefix for tool names (e.g., "memfs_read").
        """
        super().__init__(memfs)
        self.tool_prefix = tool_prefix

    def _tool_name(self, name: str) -> str:
        """Generate full tool name with prefix."""
        return f"{self.tool_prefix}_{name}"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI function calling compatible tool definitions."""
        return [
            # Read file
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("read"),
                    "description": "Read the contents of a file from MemFS. Returns file content with line numbers. Use this to view files before editing them.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to the file to read (e.g., '/memory/notes.md')",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Starting line number (1-indexed). Omit to start from beginning.",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Ending line number (inclusive). Omit to read to end.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            # Create file
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("create"),
                    "description": "Create a new file in MemFS with optional initial content. Provide a meaningful description for searchability.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path for the new file (e.g., '/memory/insights/user-preferences.md')",
                            },
                            "content": {
                                "type": "string",
                                "description": "Initial content for the file. Can be empty.",
                            },
                            "description": {
                                "type": "string",
                                "description": "A description of what this file contains. Important for search and organization.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags for filtering (e.g., ['user-preference', 'important'])",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            # Edit file
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("edit"),
                    "description": "Edit a file by replacing a specific string with a new string. The old_string must be unique in the file - include enough surrounding context to make it unique.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to the file to edit",
                            },
                            "old_string": {
                                "type": "string",
                                "description": "The exact string to replace. Must appear exactly once in the file. Include surrounding context if needed for uniqueness.",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The string to replace it with",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional: Update the file's description",
                            },
                        },
                        "required": ["path", "old_string", "new_string"],
                    },
                },
            },
            # Delete
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("delete"),
                    "description": "Delete a file or directory from MemFS. Use recursive=true to delete non-empty directories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to delete",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "If true, delete directories and all their contents. Default is false.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            # Create directory
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("mkdir"),
                    "description": "Create a new directory in MemFS. Parent directories are created automatically.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path for the new directory (e.g., '/memory/projects/alpha')",
                            },
                            "description": {
                                "type": "string",
                                "description": "A description of what this directory is for",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags for the directory",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            # List directory
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("list"),
                    "description": "List contents of a directory with metadata including descriptions and sizes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The directory path to list. Defaults to root '/'",
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth to recurse into subdirectories. Default is 1 (current directory only).",
                            },
                        },
                        "required": [],
                    },
                },
            },
            # Move/rename
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("move"),
                    "description": "Move or rename a file or directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The current path of the file or directory",
                            },
                            "destination": {
                                "type": "string",
                                "description": "The new path for the file or directory",
                            },
                        },
                        "required": ["source", "destination"],
                    },
                },
            },
            # Search
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("search"),
                    "description": "Search MemFS for files and content using semantic similarity and/or keywords. Returns relevant files and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query - can be natural language for semantic search or keywords",
                            },
                            "path": {
                                "type": "string",
                                "description": "Base path to search from. Defaults to '/' (entire MemFS)",
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["semantic", "keyword", "hybrid"],
                                "description": "Type of search: 'semantic' for meaning-based, 'keyword' for exact matches, 'hybrid' for both (default)",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of results to return. Default is 10.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional: Only search files with these tags",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            # Peek (view metadata)
            {
                "type": "function",
                "function": {
                    "name": self._tool_name("peek"),
                    "description": "View metadata about a file or folder without reading the full content. Shows description, tags, size, timestamps, and permissions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to peek at",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call and return the result as JSON string.

        Args:
            tool_name: Name of the tool (with or without prefix).
            arguments: Tool arguments.

        Returns:
            JSON string result.
        """
        # Strip prefix if present
        if tool_name.startswith(f"{self.tool_prefix}_"):
            tool_name = tool_name[len(self.tool_prefix) + 1:]

        # Map tool names to MemFS methods
        tool_map = {
            "read": self._execute_read,
            "create": self._execute_create,
            "edit": self._execute_edit,
            "delete": self._execute_delete,
            "mkdir": self._execute_mkdir,
            "list": self._execute_list,
            "move": self._execute_move,
            "search": self._execute_search,
            "peek": self._execute_peek,
        }

        if tool_name not in tool_map:
            return json.dumps({
                "status": "error",
                "message": f"Unknown tool: {tool_name}",
            })

        try:
            result = tool_map[tool_name](arguments)
            return json.dumps(result.to_dict())
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Tool execution failed: {str(e)}",
            })

    def _execute_read(self, args: dict[str, Any]):
        return self.memfs.read_file(
            path=args["path"],
            start_line=args.get("start_line"),
            end_line=args.get("end_line"),
        )

    def _execute_create(self, args: dict[str, Any]):
        return self.memfs.create_file(
            path=args["path"],
            content=args.get("content", ""),
            description=args.get("description", ""),
            tags=args.get("tags"),
        )

    def _execute_edit(self, args: dict[str, Any]):
        return self.memfs.edit_file(
            path=args["path"],
            old_string=args["old_string"],
            new_string=args["new_string"],
            description=args.get("description"),
        )

    def _execute_delete(self, args: dict[str, Any]):
        return self.memfs.delete(
            path=args["path"],
            recursive=args.get("recursive", False),
        )

    def _execute_mkdir(self, args: dict[str, Any]):
        return self.memfs.create_directory(
            path=args["path"],
            description=args.get("description", ""),
            tags=args.get("tags"),
        )

    def _execute_list(self, args: dict[str, Any]):
        return self.memfs.list_directory(
            path=args.get("path", "/"),
            max_depth=args.get("max_depth", 1),
        )

    def _execute_move(self, args: dict[str, Any]):
        return self.memfs.move(
            src=args["source"],
            dst=args["destination"],
        )

    def _execute_search(self, args: dict[str, Any]):
        return self.memfs.search(
            query=args["query"],
            path=args.get("path", "/"),
            search_type=args.get("search_type", "hybrid"),
            top_k=args.get("top_k", 10),
            tags=args.get("tags"),
        )

    def _execute_peek(self, args: dict[str, Any]):
        return self.memfs.peek(path=args["path"])


def get_openai_tools(memfs: MemFS, tool_prefix: str = "memfs") -> list[dict[str, Any]]:
    """
    Convenience function to get OpenAI-compatible tool definitions.

    Args:
        memfs: The MemFS instance.
        tool_prefix: Prefix for tool names.

    Returns:
        List of tool definitions for OpenAI function calling.
    """
    provider = OpenAIToolProvider(memfs, tool_prefix)
    return provider.get_tool_definitions()


def execute_openai_tool(
    memfs: MemFS,
    tool_name: str,
    arguments: dict[str, Any],
    tool_prefix: str = "memfs",
) -> str:
    """
    Convenience function to execute an OpenAI tool call.

    Args:
        memfs: The MemFS instance.
        tool_name: Name of the tool.
        arguments: Tool arguments.
        tool_prefix: Prefix for tool names.

    Returns:
        JSON string result.
    """
    provider = OpenAIToolProvider(memfs, tool_prefix)
    return provider.execute_tool(tool_name, arguments)
