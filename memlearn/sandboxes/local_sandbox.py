"""
Local filesystem sandbox using Python's tempfile for MemLearn.
"""

from __future__ import annotations

import os
import shutil
from tempfile import TemporaryDirectory

from memlearn.sandboxes.base import BaseSandbox, FileInfo


class LocalSandbox(BaseSandbox):
    """Local filesystem sandbox using TemporaryDirectory."""

    def __init__(self, prefix: str = "memfs-", base_dir: str | None = None):
        """
        Initialize local sandbox.

        Args:
            prefix: Prefix for the temporary directory name.
            base_dir: Base directory for creating temp dir. None uses system default.
        """
        self.prefix = prefix
        self.base_dir = base_dir
        self._temp_dir: TemporaryDirectory | None = None
        self._root_path: str | None = None

    @property
    def root_path(self) -> str:
        """Return the absolute root path of the sandbox."""
        if self._root_path is None:
            raise RuntimeError("Sandbox not initialized. Call initialize() first.")
        return self._root_path

    def initialize(self) -> None:
        """Initialize the sandbox filesystem."""
        self._temp_dir = TemporaryDirectory(prefix=self.prefix, dir=self.base_dir)
        self._root_path = self._temp_dir.name

    def cleanup(self) -> None:
        """Clean up and destroy the sandbox."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
            self._root_path = None

    def _full_path(self, path: str) -> str:
        """Convert relative path to full filesystem path."""
        # Normalize path
        if path.startswith("/"):
            path = path[1:]
        
        full = os.path.join(self.root_path, path) if path else self.root_path
        
        # Security: ensure path doesn't escape sandbox
        full = os.path.normpath(full)
        if not full.startswith(self.root_path):
            raise ValueError(f"Path '{path}' escapes sandbox")
        
        return full

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        return os.path.exists(self._full_path(path))

    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        return os.path.isfile(self._full_path(path))

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(self._full_path(path))

    def read_file(self, path: str) -> str:
        """Read file contents."""
        full_path = self._full_path(path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if os.path.isdir(full_path):
            raise IsADirectoryError(f"Path is a directory: {path}")
        
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        full_path = self._full_path(path)
        
        # Create parent directories if needed
        parent = os.path.dirname(full_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    def delete_file(self, path: str) -> None:
        """Delete a file."""
        full_path = self._full_path(path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if os.path.isdir(full_path):
            raise IsADirectoryError(f"Path is a directory: {path}")
        
        os.remove(full_path)

    def create_directory(self, path: str) -> None:
        """Create a directory (and parent directories if needed)."""
        full_path = self._full_path(path)
        os.makedirs(full_path, exist_ok=True)

    def delete_directory(self, path: str, recursive: bool = False) -> None:
        """Delete a directory."""
        full_path = self._full_path(path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not os.path.isdir(full_path):
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        if recursive:
            shutil.rmtree(full_path)
        else:
            os.rmdir(full_path)  # Will fail if not empty

    def list_directory(self, path: str) -> list[str]:
        """List contents of a directory."""
        full_path = self._full_path(path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not os.path.isdir(full_path):
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        return sorted(os.listdir(full_path))

    def move(self, src: str, dst: str) -> None:
        """Move/rename a file or directory."""
        src_full = self._full_path(src)
        dst_full = self._full_path(dst)
        
        if not os.path.exists(src_full):
            raise FileNotFoundError(f"Source not found: {src}")
        
        # Create parent directories for destination if needed
        dst_parent = os.path.dirname(dst_full)
        if dst_parent and not os.path.exists(dst_parent):
            os.makedirs(dst_parent, exist_ok=True)
        
        shutil.move(src_full, dst_full)

    def copy(self, src: str, dst: str) -> None:
        """Copy a file or directory."""
        src_full = self._full_path(src)
        dst_full = self._full_path(dst)
        
        if not os.path.exists(src_full):
            raise FileNotFoundError(f"Source not found: {src}")
        
        # Create parent directories for destination if needed
        dst_parent = os.path.dirname(dst_full)
        if dst_parent and not os.path.exists(dst_parent):
            os.makedirs(dst_parent, exist_ok=True)
        
        if os.path.isdir(src_full):
            shutil.copytree(src_full, dst_full)
        else:
            shutil.copy2(src_full, dst_full)

    def get_file_info(self, path: str) -> FileInfo:
        """Get information about a file or directory."""
        full_path = self._full_path(path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        stat = os.stat(full_path)
        
        return FileInfo(
            path=path,
            is_file=os.path.isfile(full_path),
            is_dir=os.path.isdir(full_path),
            size=stat.st_size,
            modified_time=stat.st_mtime,
        )

    def get_size(self, path: str) -> int:
        """Get size of a file in bytes."""
        full_path = self._full_path(path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Path not found: {path}")
        
        if os.path.isfile(full_path):
            return os.path.getsize(full_path)
        
        # For directories, calculate total size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(full_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return total_size

    # =========================================================================
    # Sync operations for persistence
    # =========================================================================

    def sync_from_persistent(
        self, persistent_path: str, folders: list[str]
    ) -> None:
        """
        Copy folders from persistent storage into ephemeral sandbox.

        Args:
            persistent_path: Path to persistent agent storage
                (e.g., ~/.memlearn/persistent/agents/{id}/)
            folders: Folders to sync (e.g., ["memory", "raw", "mnt"])
        """
        for folder in folders:
            src = os.path.join(persistent_path, folder)
            dst = self._full_path(folder)

            if os.path.exists(src) and os.path.isdir(src):
                # Remove existing folder in sandbox if it exists
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                # Copy from persistent to sandbox
                shutil.copytree(src, dst)

    def sync_to_persistent(
        self, persistent_path: str, folders: list[str]
    ) -> None:
        """
        Copy folders from ephemeral sandbox to persistent storage.

        Args:
            persistent_path: Path to persistent agent storage
            folders: Folders to sync (e.g., ["memory", "raw", "mnt"])
        """
        # Ensure persistent path exists
        os.makedirs(persistent_path, exist_ok=True)

        for folder in folders:
            src = self._full_path(folder)
            dst = os.path.join(persistent_path, folder)

            if os.path.exists(src) and os.path.isdir(src):
                # Remove existing folder in persistent storage if it exists
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                # Copy from sandbox to persistent
                shutil.copytree(src, dst)

    def clear_directory_contents(self, path: str) -> None:
        """
        Clear all contents of a directory without deleting the directory itself.

        Args:
            path: Relative path within the sandbox.
        """
        full_path = self._full_path(path)

        if not os.path.exists(full_path):
            return  # Nothing to clear

        if not os.path.isdir(full_path):
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Remove all contents
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
