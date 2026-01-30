"""
LangChain-compatible tool definitions for MemFS operations.

Provides tools that can be used with LangChain agents via:
- LangChainToolProvider class for direct integration
- get_langchain_tools() convenience function
- Individual tool functions with @tool decorators for custom usage

Example usage:
    from memlearn import MemFS
    from memlearn.tools.langchain_tools import LangChainToolProvider, get_langchain_tools

    # Using the provider class
    with MemFS.for_agent("my-agent") as memfs:
        provider = LangChainToolProvider(memfs)
        tools = provider.get_tools()

        # Use with LangChain agent
        agent = create_react_agent(llm, tools, prompt)

    # Or use the convenience function
    tools = get_langchain_tools(memfs)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field

from memlearn.tools.base import BaseToolProvider

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from memlearn.memfs import MemFS


class ReadFileInput(BaseModel):
    """Input schema for reading a file from MemFS."""

    path: str = Field(
        description="The path to the file to read (e.g., '/memory/notes.md')"
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Starting line number (1-indexed). Omit to start from beginning.",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="Ending line number (inclusive). Omit to read to end.",
    )


class CreateFileInput(BaseModel):
    """Input schema for creating a file in MemFS."""

    path: str = Field(
        description="The path for the new file (e.g., '/memory/insights/user-preferences.md')"
    )
    content: str = Field(
        default="", description="Initial content for the file. Can be empty."
    )
    description: str = Field(
        default="",
        description="A description of what this file contains. Important for search and organization.",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Optional tags for filtering (e.g., ['user-preference', 'important'])",
    )


class EditFileInput(BaseModel):
    """Input schema for editing a file in MemFS."""

    path: str = Field(description="The path to the file to edit")
    old_string: str = Field(
        description="The exact string to replace. Must appear exactly once in the file. Include surrounding context if needed for uniqueness."
    )
    new_string: str = Field(description="The string to replace it with")
    description: Optional[str] = Field(
        default=None, description="Optional: Update the file's description"
    )


class DeleteInput(BaseModel):
    """Input schema for deleting a file or directory from MemFS."""

    path: str = Field(description="The path to delete")
    recursive: bool = Field(
        default=False,
        description="If true, delete directories and all their contents. Default is false.",
    )


class MkdirInput(BaseModel):
    """Input schema for creating a directory in MemFS."""

    path: str = Field(
        description="The path for the new directory (e.g., '/memory/projects/alpha')"
    )
    description: str = Field(
        default="", description="A description of what this directory is for"
    )
    tags: Optional[list[str]] = Field(
        default=None, description="Optional tags for the directory"
    )


class ListInput(BaseModel):
    """Input schema for listing directory contents in MemFS."""

    path: str = Field(
        default="/", description="The directory path to list. Defaults to root '/'"
    )
    max_depth: int = Field(
        default=1,
        description="Maximum depth to recurse into subdirectories. Default is 1 (current directory only).",
    )


class MoveInput(BaseModel):
    """Input schema for moving/renaming a file or directory in MemFS."""

    source: str = Field(description="The current path of the file or directory")
    destination: str = Field(description="The new path for the file or directory")


class SearchInput(BaseModel):
    """Input schema for searching MemFS."""

    query: str = Field(
        description="The search query - can be natural language for semantic search or keywords"
    )
    path: str = Field(
        default="/",
        description="Base path to search from. Defaults to '/' (entire MemFS)",
    )
    search_type: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="Type of search: 'semantic' for meaning-based, 'keyword' for exact matches, 'hybrid' for both (default)",
    )
    top_k: int = Field(
        default=10, description="Maximum number of results to return. Default is 10."
    )
    tags: Optional[list[str]] = Field(
        default=None, description="Optional: Only search files with these tags"
    )


class PeekInput(BaseModel):
    """Input schema for peeking at metadata in MemFS."""

    path: str = Field(description="The path to peek at")


class CompactInput(BaseModel):
    """Input schema for compacting conversation history."""

    summary: str = Field(
        description="A comprehensive summary of the conversation and progress so far. Include key decisions, important context, and current state."
    )
    preserve_last_n: int = Field(
        default=10,
        description="Number of recent messages to keep uncompacted. Default is 10.",
    )


class FindInput(BaseModel):
    """Input schema for finding files by name pattern in MemFS."""

    pattern: str = Field(
        description="Pattern to match against file names. For glob: *.md, test_*.py, **/*.json. For regex: .*\\.md$. For exact: README.md"
    )
    path: str = Field(
        default="/",
        description="Base directory to search from. Defaults to '/' (entire MemFS).",
    )
    match_type: Literal["glob", "regex", "exact"] = Field(
        default="glob",
        description="Pattern matching type. 'glob' for shell-style wildcards (default), 'regex' for regular expressions, 'exact' for exact filename match.",
    )
    include_dirs: bool = Field(
        default=False,
        description="If true, also include matching directories in results. Default is false (files only).",
    )
    max_results: int = Field(
        default=50,
        description="Maximum results per page (default 50). Use -1 for unlimited (caution: may be large).",
    )
    offset: int = Field(
        default=0,
        description="Number of results to skip for pagination. Use 'next_offset' from previous response to get next page.",
    )
    sort_by: Literal["path", "name", "size", "modified", "created"] = Field(
        default="path",
        description="Sort results by: 'path' (default), 'name', 'size', 'modified', or 'created'.",
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="asc",
        description="Sort order: 'asc' (ascending, default) or 'desc' (descending).",
    )


class BashInput(BaseModel):
    """Input schema for executing bash commands in MemFS sandbox."""

    command: str = Field(
        description="The bash command to execute (e.g., 'ls -la', 'cat file.txt | grep pattern', 'python script.py')"
    )
    timeout: int = Field(
        default=30,
        description="Maximum execution time in seconds. Default is 30. Maximum is 300.",
    )
    working_dir: str = Field(
        default="/",
        description="Working directory relative to MemFS root (e.g., '/memory/scripts'). Default is '/' (MemFS root).",
    )


class UndoInput(BaseModel):
    """Input schema for undoing filesystem changes."""

    count: int = Field(
        default=1,
        description="Number of changes to undo. Use -1 to undo all changes in the current session. Default is 1.",
    )


class LangChainToolProvider(BaseToolProvider):
    """LangChain-compatible tool provider for MemFS."""

    def __init__(
        self,
        memfs: MemFS,
        tool_prefix: str = "memfs",
        enable_bash: bool = False,
        read_only: bool = False,
    ):
        """
        Initialize the LangChain tool provider.

        Args:
            memfs: The MemFS instance to operate on.
            tool_prefix: Prefix for tool names (e.g., "memfs_read").
            enable_bash: Whether to enable the bash command execution tool.
                WARNING: This allows arbitrary command execution within the
                MemFS sandbox. Use with caution. Defaults to False.
            read_only: If True, only provides read-only tools (read, list, search,
                peek, find). Write tools (create, edit, delete, mkdir, move, compact)
                are excluded. If bash is enabled, its description instructs the LLM
                not to modify the filesystem.
        """
        super().__init__(memfs)
        self.tool_prefix = tool_prefix
        self.enable_bash = enable_bash
        self.read_only = read_only

    def _tool_name(self, name: str) -> str:
        """Generate full tool name with prefix."""
        return f"{self.tool_prefix}_{name}"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions as dictionaries (for compatibility with base class).

        For LangChain usage, prefer get_tools() which returns actual Tool objects.
        """
        tools = self.get_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema,
            }
            for tool in tools
        ]

    def get_tools(self) -> list[BaseTool]:
        """
        Get LangChain Tool objects for use with agents.

        In read-only mode, only returns read-only tools: read, list, search, peek, find.
        Write tools (create, edit, delete, mkdir, move, compact) are excluded.

        Returns:
            List of LangChain StructuredTool instances.
        """
        from langchain_core.tools import StructuredTool

        # Read-only tools (always included)
        tools = [
            StructuredTool.from_function(
                func=self._read_file,
                name=self._tool_name("read"),
                description="Read the contents of a file from MemFS. Returns file content with line numbers. Use this to view files before editing them.",
                args_schema=ReadFileInput,
            ),
            StructuredTool.from_function(
                func=self._list_dir,
                name=self._tool_name("list"),
                description="List contents of a directory with metadata including descriptions and sizes.",
                args_schema=ListInput,
            ),
            StructuredTool.from_function(
                func=self._search,
                name=self._tool_name("search"),
                description="Search MemFS for files and content using semantic similarity and/or keywords. Returns relevant files and snippets.",
                args_schema=SearchInput,
            ),
            StructuredTool.from_function(
                func=self._peek,
                name=self._tool_name("peek"),
                description="View metadata about a file or folder without reading the full content. Shows description, tags, size, timestamps, and permissions.",
                args_schema=PeekInput,
            ),
            StructuredTool.from_function(
                func=self._find,
                name=self._tool_name("find"),
                description="Find files by name pattern with pagination and metadata. Use glob patterns (*.md, test_*.py, **/*.json), regex, or exact matches. Returns files with metadata (size, type, description, tags, timestamps) in an LLM-friendly format with automatic pagination to avoid context overflow.",
                args_schema=FindInput,
            ),
        ]

        # Write tools (only in read-write mode)
        if not self.read_only:
            tools.extend(
                [
                    StructuredTool.from_function(
                        func=self._create_file,
                        name=self._tool_name("create"),
                        description="Create a new file in MemFS with optional initial content. Provide a meaningful description for searchability.",
                        args_schema=CreateFileInput,
                    ),
                    StructuredTool.from_function(
                        func=self._edit_file,
                        name=self._tool_name("edit"),
                        description="Edit a file by replacing a specific string with a new string. The old_string must be unique in the file - include enough surrounding context to make it unique.",
                        args_schema=EditFileInput,
                    ),
                    StructuredTool.from_function(
                        func=self._delete,
                        name=self._tool_name("delete"),
                        description="Delete a file or directory from MemFS. Use recursive=true to delete non-empty directories.",
                        args_schema=DeleteInput,
                    ),
                    StructuredTool.from_function(
                        func=self._mkdir,
                        name=self._tool_name("mkdir"),
                        description="Create a new directory in MemFS. Parent directories are created automatically.",
                        args_schema=MkdirInput,
                    ),
                    StructuredTool.from_function(
                        func=self._move,
                        name=self._tool_name("move"),
                        description="Move or rename a file or directory.",
                        args_schema=MoveInput,
                    ),
                    StructuredTool.from_function(
                        func=self._compact,
                        name=self._tool_name("compact"),
                        description="Compact the conversation history by summarizing older messages. Use when the context window is getting full. The full history is preserved in MemFS for observability. Returns compacted messages to use for the next LLM call.",
                        args_schema=CompactInput,
                    ),
                    StructuredTool.from_function(
                        func=self._undo,
                        name=self._tool_name("undo"),
                        description="Undo recent filesystem changes within the current session. Use this to revert mistakes or unwanted changes. Specify count=1 to undo the last change, count=N to undo N changes, or count=-1 to undo all changes in this session.",
                        args_schema=UndoInput,
                    ),
                ]
            )

        # Bash tool (with different description in read-only mode)
        if self.enable_bash:
            if self.read_only:
                bash_description = (
                    "Execute a bash command within the MemFS sandbox for READ-ONLY operations. "
                    "The command runs with the MemFS root as the working directory. "
                    "IMPORTANT: This MemFS instance is in READ-ONLY mode. You MUST NOT execute any commands that "
                    "modify, create, delete, or write files. Only use commands that read or inspect data "
                    "(e.g., 'cat', 'ls', 'grep', 'find', 'head', 'tail', 'wc'). "
                    "Do NOT use commands like 'rm', 'mv', 'cp', 'mkdir', 'touch', 'echo >', 'sed -i', etc."
                )
            else:
                bash_description = (
                    "Execute a bash command within the MemFS sandbox. The command runs with the MemFS root "
                    "as the working directory. Use this for tasks like running scripts, processing files, "
                    "or performing operations that require shell commands. Commands are executed in a sandboxed environment."
                )

            tools.append(
                StructuredTool.from_function(
                    func=self._bash,
                    name=self._tool_name("bash"),
                    description=bash_description,
                    args_schema=BashInput,
                )
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call and return the result as JSON string.

        In read-only mode, write operations return an error.

        Args:
            tool_name: Name of the tool (with or without prefix).
            arguments: Tool arguments.

        Returns:
            JSON string result.
        """
        # Strip prefix if present
        if tool_name.startswith(f"{self.tool_prefix}_"):
            tool_name = tool_name[len(self.tool_prefix) + 1 :]

        # Define write tools that are blocked in read-only mode
        write_tools = {"create", "edit", "delete", "mkdir", "move", "compact", "undo"}

        # Block write operations in read-only mode
        if self.read_only and tool_name in write_tools:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool '{tool_name}' is not available in read-only mode. Only read, list, search, peek, and find are allowed.",
                }
            )

        # Read-only tools
        tool_map = {
            "read": self._read_file,
            "list": self._list_dir,
            "search": self._search,
            "peek": self._peek,
            "find": self._find,
        }

        # Add write tools only in read-write mode
        if not self.read_only:
            tool_map.update(
                {
                    "create": self._create_file,
                    "edit": self._edit_file,
                    "delete": self._delete,
                    "mkdir": self._mkdir,
                    "move": self._move,
                    "compact": self._compact,
                    "undo": self._undo,
                }
            )

        if self.enable_bash:
            tool_map["bash"] = self._bash

        if tool_name not in tool_map:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Unknown tool: {tool_name}",
                }
            )

        try:
            result = tool_map[tool_name](**arguments)
            return result
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool execution failed: {str(e)}",
                }
            )

    def _read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read a file from MemFS."""
        result = self.memfs.read_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
        )
        return json.dumps(result.to_dict())

    def _create_file(
        self,
        path: str,
        content: str = "",
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> str:
        """Create a new file in MemFS."""
        result = self.memfs.create_file(
            path=path,
            content=content,
            description=description,
            tags=tags,
        )
        return json.dumps(result.to_dict())

    def _edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        description: Optional[str] = None,
    ) -> str:
        """Edit a file in MemFS using string replacement."""
        result = self.memfs.edit_file(
            path=path,
            old_string=old_string,
            new_string=new_string,
            description=description,
        )
        return json.dumps(result.to_dict())

    def _delete(
        self,
        path: str,
        recursive: bool = False,
    ) -> str:
        """Delete a file or directory from MemFS."""
        result = self.memfs.delete(
            path=path,
            recursive=recursive,
        )
        return json.dumps(result.to_dict())

    def _mkdir(
        self,
        path: str,
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> str:
        """Create a directory in MemFS."""
        result = self.memfs.create_directory(
            path=path,
            description=description,
            tags=tags,
        )
        return json.dumps(result.to_dict())

    def _list_dir(
        self,
        path: str = "/",
        max_depth: int = 1,
    ) -> str:
        """List contents of a directory in MemFS."""
        result = self.memfs.list_directory(
            path=path,
            max_depth=max_depth,
        )
        return json.dumps(result.to_dict())

    def _move(
        self,
        source: str,
        destination: str,
    ) -> str:
        """Move or rename a file/directory in MemFS."""
        result = self.memfs.move(
            src=source,
            dst=destination,
        )
        return json.dumps(result.to_dict())

    def _search(
        self,
        query: str,
        path: str = "/",
        search_type: str = "hybrid",
        top_k: int = 10,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Search MemFS for files and content."""
        result = self.memfs.search(
            query=query,
            path=path,
            search_type=search_type,
            top_k=top_k,
            tags=tags,
        )
        return json.dumps(result.to_dict())

    def _peek(self, path: str) -> str:
        """View metadata about a file or folder."""
        result = self.memfs.peek(path=path)
        return json.dumps(result.to_dict())

    def _compact(
        self,
        summary: str,
        preserve_last_n: int = 10,
    ) -> str:
        """Compact conversation history."""
        result = self.memfs.compact_conversation(
            summary=summary,
            preserve_last_n=preserve_last_n,
        )
        return json.dumps(result.to_dict())

    def _find(
        self,
        pattern: str,
        path: str = "/",
        match_type: str = "glob",
        include_dirs: bool = False,
        max_results: int = 50,
        offset: int = 0,
        sort_by: str = "path",
        sort_order: str = "asc",
    ) -> str:
        """Find files by name pattern."""
        result = self.memfs.find(
            pattern=pattern,
            path=path,
            match_type=match_type,
            include_dirs=include_dirs,
            max_results=max_results,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return json.dumps(result.to_dict())

    def _undo(
        self,
        count: int = 1,
    ) -> str:
        """Undo recent filesystem changes."""
        result = self.memfs.undo(count=count)
        return json.dumps(result.to_dict())

    def _bash(
        self,
        command: str,
        timeout: int = 30,
        working_dir: str = "/",
    ) -> str:
        """Execute a bash command in the MemFS sandbox."""
        import os
        import subprocess

        from memlearn.types import ToolResult

        if not self.enable_bash:
            result = ToolResult(
                status="error",
                message="Bash execution is not enabled. Set enable_bash=True to enable.",
            )
            return json.dumps(result.to_dict())

        if not command:
            result = ToolResult(
                status="error",
                message="No command provided.",
            )
            return json.dumps(result.to_dict())

        # Limit timeout to reasonable bounds
        timeout = min(max(timeout, 1), 300)

        # Get working directory within sandbox
        if working_dir.startswith("/"):
            working_dir = working_dir[1:]

        # Get sandbox root path
        sandbox_root = self.memfs.sandbox.root_path

        # Resolve working directory
        if working_dir:
            cwd = os.path.normpath(os.path.join(sandbox_root, working_dir))
        else:
            cwd = sandbox_root

        # Security: ensure working directory doesn't escape sandbox
        if not cwd.startswith(sandbox_root):
            result = ToolResult(
                status="error",
                message=f"Working directory escapes sandbox: {working_dir}",
            )
            return json.dumps(result.to_dict())

        # Ensure working directory exists
        if not os.path.isdir(cwd):
            result = ToolResult(
                status="error",
                message=f"Working directory does not exist: {working_dir}",
            )
            return json.dumps(result.to_dict())

        try:
            # Execute the command
            proc_result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "HOME": sandbox_root,
                    "MEMFS_ROOT": sandbox_root,
                },
            )

            # Combine stdout and stderr
            output = ""
            if proc_result.stdout:
                output += proc_result.stdout
            if proc_result.stderr:
                if output:
                    output += "\n--- stderr ---\n"
                output += proc_result.stderr

            # Truncate very long output
            max_output_len = 50000
            if len(output) > max_output_len:
                output = (
                    output[:max_output_len]
                    + f"\n... [truncated, {len(output) - max_output_len} chars omitted]"
                )

            result = ToolResult(
                status="success" if proc_result.returncode == 0 else "error",
                message=f"Command exited with code {proc_result.returncode}",
                data={
                    "exit_code": proc_result.returncode,
                    "output": output,
                    "command": command,
                    "working_dir": working_dir or "/",
                },
            )
            return json.dumps(result.to_dict())

        except subprocess.TimeoutExpired:
            result = ToolResult(
                status="error",
                message=f"Command timed out after {timeout} seconds.",
                data={"command": command, "timeout": timeout},
            )
            return json.dumps(result.to_dict())
        except Exception as e:
            result = ToolResult(
                status="error",
                message=f"Failed to execute command: {str(e)}",
                data={"command": command},
            )
            return json.dumps(result.to_dict())


def get_langchain_tools(
    memfs: MemFS,
    tool_prefix: str = "memfs",
    enable_bash: bool = False,
    read_only: bool | None = None,
) -> list[BaseTool]:
    """
    Convenience function to get LangChain-compatible tools for MemFS.

    Args:
        memfs: The MemFS instance.
        tool_prefix: Prefix for tool names.
        enable_bash: Whether to enable the bash command execution tool.
            WARNING: This allows arbitrary command execution within the
            MemFS sandbox. Use with caution. Defaults to False.
        read_only: If True, only returns read-only tools. If None (default),
            inherits from memfs.read_only.

    Returns:
        List of LangChain StructuredTool instances for use with agents.

    Example:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_react_agent, AgentExecutor
        from memlearn import MemFS
        from memlearn.tools.langchain_tools import get_langchain_tools

        llm = ChatOpenAI(model="gpt-4")

        with MemFS.for_agent("my-agent") as memfs:
            tools = get_langchain_tools(memfs)

            # Create and run agent
            agent = create_react_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools)
            result = executor.invoke({"input": "Save a note about Python"})
    """
    # Use memfs.read_only if not explicitly specified
    if read_only is None:
        read_only = memfs.read_only
    provider = LangChainToolProvider(
        memfs, tool_prefix, enable_bash=enable_bash, read_only=read_only
    )
    return provider.get_tools()


def execute_langchain_tool(
    memfs: MemFS,
    tool_name: str,
    arguments: dict[str, Any],
    tool_prefix: str = "memfs",
    enable_bash: bool = False,
    read_only: bool | None = None,
) -> str:
    """
    Convenience function to execute a LangChain tool call.

    Args:
        memfs: The MemFS instance.
        tool_name: Name of the tool.
        arguments: Tool arguments.
        tool_prefix: Prefix for tool names.
        enable_bash: Whether to enable the bash command execution tool.
        read_only: If True, blocks write operations. If None (default),
            inherits from memfs.read_only.

    Returns:
        JSON string result.
    """
    # Use memfs.read_only if not explicitly specified
    if read_only is None:
        read_only = memfs.read_only
    provider = LangChainToolProvider(
        memfs, tool_prefix, enable_bash=enable_bash, read_only=read_only
    )
    return provider.execute_tool(tool_name, arguments)
