"""
Base classes for MemLearn tool definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from memlearn.memfs import MemFS


class BaseToolProvider(ABC):
    """Abstract base class for tool providers that generate agent-specific tool definitions."""

    def __init__(self, memfs: MemFS):
        """
        Initialize the tool provider.

        Args:
            memfs: The MemFS instance to operate on.
        """
        self.memfs = memfs

    @abstractmethod
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions in the format expected by the target agent framework.

        Returns:
            List of tool definitions.
        """
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            String result to return to the agent.
        """
        pass
