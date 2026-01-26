"""
Abstract base class for MemLearn filesystem sandboxes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Information about a file or directory."""

    path: str
    is_file: bool
    is_dir: bool
    size: int
    modified_time: float


class BaseSandbox(ABC):
    """Abstract base class for ephemeral filesystem sandboxes."""

    @property
    @abstractmethod
    def root_path(self) -> str:
        """Return the absolute root path of the sandbox."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the sandbox filesystem."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up and destroy the sandbox."""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        pass

    @abstractmethod
    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        pass

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        pass

    @abstractmethod
    def read_file(self, path: str) -> str:
        """
        Read file contents.

        Args:
            path: Relative path within the sandbox.

        Returns:
            File contents as string.

        Raises:
            FileNotFoundError: If file doesn't exist.
            IsADirectoryError: If path is a directory.
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """
        Write content to a file.

        Args:
            path: Relative path within the sandbox.
            content: Content to write.
        """
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """
        Delete a file.

        Args:
            path: Relative path within the sandbox.

        Raises:
            FileNotFoundError: If file doesn't exist.
            IsADirectoryError: If path is a directory.
        """
        pass

    @abstractmethod
    def create_directory(self, path: str) -> None:
        """
        Create a directory (and parent directories if needed).

        Args:
            path: Relative path within the sandbox.
        """
        pass

    @abstractmethod
    def delete_directory(self, path: str, recursive: bool = False) -> None:
        """
        Delete a directory.

        Args:
            path: Relative path within the sandbox.
            recursive: If True, delete contents recursively.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            NotADirectoryError: If path is not a directory.
            OSError: If directory is not empty and recursive is False.
        """
        pass

    @abstractmethod
    def list_directory(self, path: str) -> list[str]:
        """
        List contents of a directory.

        Args:
            path: Relative path within the sandbox.

        Returns:
            List of filenames in the directory.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            NotADirectoryError: If path is not a directory.
        """
        pass

    @abstractmethod
    def move(self, src: str, dst: str) -> None:
        """
        Move/rename a file or directory.

        Args:
            src: Source path.
            dst: Destination path.

        Raises:
            FileNotFoundError: If source doesn't exist.
        """
        pass

    @abstractmethod
    def copy(self, src: str, dst: str) -> None:
        """
        Copy a file or directory.

        Args:
            src: Source path.
            dst: Destination path.

        Raises:
            FileNotFoundError: If source doesn't exist.
        """
        pass

    @abstractmethod
    def get_file_info(self, path: str) -> FileInfo:
        """
        Get information about a file or directory.

        Args:
            path: Relative path within the sandbox.

        Returns:
            FileInfo object with path information.

        Raises:
            FileNotFoundError: If path doesn't exist.
        """
        pass

    @abstractmethod
    def get_size(self, path: str) -> int:
        """
        Get size of a file in bytes.

        Args:
            path: Relative path within the sandbox.

        Returns:
            Size in bytes.
        """
        pass

    def resolve_path(self, path: str) -> str:
        """
        Resolve a relative path to absolute path within sandbox.

        Args:
            path: Relative path (may start with /).

        Returns:
            Absolute path within the sandbox.
        """
        # Remove leading slash for joining
        if path.startswith("/"):
            path = path[1:]
        return f"{self.root_path}/{path}" if path else self.root_path
