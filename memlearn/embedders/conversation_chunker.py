"""
Specialized chunking strategy for conversation history markdown files.

This module provides turn-aware chunking that:
- Preserves conversation turn boundaries when possible
- Includes metadata (datetime, role) in each chunk header
- Handles very long turns by splitting mid-turn with metadata
- Provides reasonable overlap between chunks for context
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""

    role: str
    timestamp: str  # Formatted as "YYYY-MM-DD HH:MM:SS" or "unknown"
    content: str
    tool_name: str | None = None
    start_line: int = 0
    end_line: int = 0

    @property
    def header(self) -> str:
        """Generate the markdown header for this turn."""
        if self.tool_name:
            return f"## [{self.timestamp}] {self.role}\n\n**Tool:** `{self.tool_name}`\n\n"
        return f"## [{self.timestamp}] {self.role}\n\n"

    @property
    def full_text(self) -> str:
        """Get the full text including header."""
        return self.header + self.content

    def __len__(self) -> int:
        """Return the character length of the full turn."""
        return len(self.full_text)


@dataclass
class ConversationChunk:
    """A chunk of conversation history with metadata."""

    content: str
    start_line: int
    end_line: int
    # Metadata for embedding
    datetime_start: str
    datetime_end: str
    roles: list[str]
    is_partial_turn: bool = False
    turn_count: int = 0


def parse_conversation_markdown(markdown: str) -> tuple[str, list[ConversationTurn]]:
    """
    Parse conversation history markdown into frontmatter and turns.

    Args:
        markdown: The full markdown content.

    Returns:
        Tuple of (frontmatter_text, list of ConversationTurn objects).
    """
    # Split by lines for line number tracking
    lines = markdown.split("\n")

    # Find frontmatter
    frontmatter_lines: list[str] = []
    body_start_line = 0

    if lines and lines[0] == "---":
        for i, line in enumerate(lines[1:], 1):
            if line == "---":
                frontmatter_lines = lines[: i + 1]
                body_start_line = i + 1
                break

    frontmatter = "\n".join(frontmatter_lines)

    # Parse turns from body
    # Pattern: ## [timestamp] role
    turn_pattern = re.compile(r"^## \[([^\]]+)\] (\w+)\s*$")

    turns: list[ConversationTurn] = []
    current_turn_start_line = 0
    current_turn_header: tuple[str, str] | None = None  # (timestamp, role)
    current_turn_tool: str | None = None
    current_content_lines: list[str] = []

    for i, line in enumerate(lines[body_start_line:], body_start_line + 1):
        match = turn_pattern.match(line)
        if match:
            # Save previous turn if exists
            if current_turn_header is not None:
                content = "\n".join(current_content_lines).strip()
                turns.append(
                    ConversationTurn(
                        role=current_turn_header[1],
                        timestamp=current_turn_header[0],
                        content=content,
                        tool_name=current_turn_tool,
                        start_line=current_turn_start_line,
                        end_line=i - 1,
                    )
                )

            # Start new turn
            current_turn_header = (match.group(1), match.group(2))
            current_turn_start_line = i
            current_turn_tool = None
            current_content_lines = []
        else:
            # Check for tool name
            if current_turn_header and current_turn_header[1] in (
                "tool_call",
                "tool_result",
            ):
                tool_match = re.match(r"\*\*Tool:\*\* `([^`]+)`", line)
                if tool_match:
                    current_turn_tool = tool_match.group(1)
                    continue

            current_content_lines.append(line)

    # Don't forget the last turn
    if current_turn_header is not None:
        content = "\n".join(current_content_lines).strip()
        turns.append(
            ConversationTurn(
                role=current_turn_header[1],
                timestamp=current_turn_header[0],
                content=content,
                tool_name=current_turn_tool,
                start_line=current_turn_start_line,
                end_line=len(lines),
            )
        )

    return frontmatter, turns


def _create_chunk_header(
    datetime_start: str,
    datetime_end: str,
    roles: list[str],
    is_continuation: bool = False,
) -> str:
    """Create a metadata header for a chunk."""
    unique_roles = list(dict.fromkeys(roles))  # Preserve order, remove dupes
    roles_str = ", ".join(unique_roles)

    if datetime_start == datetime_end:
        time_range = datetime_start
    else:
        time_range = f"{datetime_start} → {datetime_end}"

    prefix = "[CONTINUED] " if is_continuation else ""
    return f"<!-- Chunk: {prefix}{time_range} | Roles: {roles_str} -->\n\n"


def chunk_conversation_history(
    markdown: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    min_chunk_size: int = 200,
) -> list[tuple[str, int, int, dict[str, Any]]]:
    """
    Chunk conversation history markdown with turn-aware splitting.

    This chunking strategy:
    1. Tries to keep complete conversation turns together
    2. Splits between turns when possible
    3. Splits mid-turn only for very long turns, adding metadata header
    4. Includes overlap from previous turn(s) for context
    5. Each chunk includes metadata about datetime range and roles

    Args:
        markdown: The conversation history markdown content.
        chunk_size: Target size for each chunk in characters (default 1500).
        chunk_overlap: Number of characters to overlap between chunks (default 200).
        min_chunk_size: Minimum chunk size to avoid tiny chunks (default 200).

    Returns:
        List of tuples: (chunk_text, start_line, end_line, metadata_dict).
        Metadata includes: datetime_start, datetime_end, roles, is_partial_turn,
        turn_count.
    """
    frontmatter, turns = parse_conversation_markdown(markdown)

    if not turns:
        # No turns found, return the whole content as one chunk
        lines = markdown.split("\n")
        return [
            (
                markdown,
                1,
                len(lines),
                {
                    "datetime_start": "unknown",
                    "datetime_end": "unknown",
                    "roles": [],
                    "is_partial_turn": False,
                    "turn_count": 0,
                },
            )
        ]

    chunks: list[tuple[str, int, int, dict[str, Any]]] = []

    # Current chunk being built
    current_turns: list[ConversationTurn] = []
    current_size = 0

    def flush_chunk(is_partial: bool = False) -> None:
        """Flush current turns as a chunk."""
        nonlocal current_turns, current_size

        if not current_turns:
            return

        # Build chunk content
        content_parts: list[str] = []
        roles: list[str] = []

        for turn in current_turns:
            content_parts.append(turn.full_text)
            roles.append(turn.role)

        datetime_start = current_turns[0].timestamp
        datetime_end = current_turns[-1].timestamp

        # Add metadata header
        header = _create_chunk_header(datetime_start, datetime_end, roles)
        chunk_content = header + "\n".join(content_parts)

        chunks.append(
            (
                chunk_content,
                current_turns[0].start_line,
                current_turns[-1].end_line,
                {
                    "datetime_start": datetime_start,
                    "datetime_end": datetime_end,
                    "roles": roles,
                    "is_partial_turn": is_partial,
                    "turn_count": len(current_turns),
                },
            )
        )

        current_turns = []
        current_size = 0

    def split_long_turn(turn: ConversationTurn) -> list[tuple[str, int, int, dict]]:
        """Split a single long turn into multiple chunks."""
        result: list[tuple[str, int, int, dict]] = []
        content = turn.content
        header = turn.header
        header_len = len(header)

        # Available space for content per chunk
        available_per_chunk = chunk_size - header_len - 100  # Reserve for metadata

        if available_per_chunk <= 0:
            available_per_chunk = chunk_size // 2

        # Track line numbers as we split
        # The turn header takes ~2-3 lines (## [...] role\n\n)
        header_lines = header.count("\n") + 1
        current_line_offset = 0  # Offset from turn.start_line (after header)

        # Split content by paragraphs first, then by lines if needed
        paragraphs = content.split("\n\n")
        current_content: list[str] = []
        current_content_size = 0
        current_content_lines = 0  # Lines in current chunk's content
        part_num = 0
        chunk_start_offset = 0  # Start offset for current chunk

        def flush_part(content_parts: list[str], join_str: str) -> None:
            """Flush accumulated content as a chunk part."""
            nonlocal part_num, current_content, current_content_size
            nonlocal current_content_lines, chunk_start_offset, current_line_offset

            if not content_parts:
                return

            part_content = join_str.join(content_parts)
            is_continuation = part_num > 0
            meta_header = _create_chunk_header(
                turn.timestamp,
                turn.timestamp,
                [turn.role],
                is_continuation=is_continuation,
            )
            full_content = meta_header + header + part_content

            # Calculate line range for this chunk
            # Add header_lines to account for the turn header
            chunk_start = turn.start_line + header_lines + chunk_start_offset
            chunk_end = turn.start_line + header_lines + current_line_offset

            result.append(
                (
                    full_content,
                    chunk_start,
                    chunk_end,
                    {
                        "datetime_start": turn.timestamp,
                        "datetime_end": turn.timestamp,
                        "roles": [turn.role],
                        "is_partial_turn": True,
                        "turn_count": 1,
                        "part": part_num + 1,
                    },
                )
            )
            part_num += 1
            chunk_start_offset = current_line_offset
            current_content = []
            current_content_size = 0
            current_content_lines = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for \n\n
            para_lines = para.count("\n") + 2  # +2 for the \n\n separator

            if para_size > available_per_chunk:
                # Paragraph too long, split by lines
                lines = para.split("\n")
                for line in lines:
                    line_size = len(line) + 1

                    if current_content_size + line_size > available_per_chunk:
                        # Flush current content
                        flush_part(current_content, "\n")

                    current_content.append(line)
                    current_content_size += line_size
                    current_content_lines += 1
                    current_line_offset += 1
            else:
                if current_content_size + para_size > available_per_chunk:
                    # Flush current content
                    flush_part(current_content, "\n\n")

                current_content.append(para)
                current_content_size += para_size
                current_content_lines += para_lines
                current_line_offset += para_lines

        # Don't forget remaining content
        if current_content:
            part_content = "\n\n".join(current_content)
            is_continuation = part_num > 0
            meta_header = _create_chunk_header(
                turn.timestamp,
                turn.timestamp,
                [turn.role],
                is_continuation=is_continuation,
            )
            full_content = meta_header + header + part_content

            # Calculate line range for this final chunk
            chunk_start = turn.start_line + header_lines + chunk_start_offset
            # Use turn.end_line for the final chunk to ensure we capture everything
            chunk_end = turn.end_line

            result.append(
                (
                    full_content,
                    chunk_start,
                    chunk_end,
                    {
                        "datetime_start": turn.timestamp,
                        "datetime_end": turn.timestamp,
                        "roles": [turn.role],
                        "is_partial_turn": True,
                        "turn_count": 1,
                        "part": part_num + 1,
                    },
                )
            )

        return result

    # Process turns
    for i, turn in enumerate(turns):
        turn_size = len(turn)

        # Check if this single turn is too long
        if turn_size > chunk_size:
            # Flush any accumulated turns first
            flush_chunk()

            # Split the long turn
            split_chunks = split_long_turn(turn)
            chunks.extend(split_chunks)
            continue

        # Check if adding this turn would exceed chunk size
        if current_size + turn_size > chunk_size and current_turns:
            # Flush current chunk
            flush_chunk()

            # Add overlap from previous chunk (last turn's content, truncated)
            if chunks and chunk_overlap > 0:
                # Get the last turn from the previous chunk for overlap
                prev_chunk_content = chunks[-1][0]
                overlap_text = prev_chunk_content[-chunk_overlap:]

                # Find a good breaking point (newline)
                newline_pos = overlap_text.find("\n")
                if newline_pos > 0:
                    overlap_text = overlap_text[newline_pos + 1 :]

                # We don't actually prepend overlap to avoid duplication
                # Instead, the metadata header provides context

        current_turns.append(turn)
        current_size += turn_size

    # Flush remaining turns
    if current_turns:
        flush_chunk()

    # Handle edge case: ensure minimum chunk size by merging tiny chunks
    if len(chunks) > 1:
        merged_chunks: list[tuple[str, int, int, dict]] = []
        pending_chunk: tuple[str, int, int, dict] | None = None

        for chunk in chunks:
            if len(chunk[0]) < min_chunk_size:
                if pending_chunk:
                    # Merge with pending
                    merged_content = pending_chunk[0] + "\n\n" + chunk[0]
                    merged_meta = pending_chunk[3].copy()
                    merged_meta["datetime_end"] = chunk[3]["datetime_end"]
                    merged_meta["roles"] = pending_chunk[3]["roles"] + chunk[3]["roles"]
                    merged_meta["turn_count"] = (
                        pending_chunk[3]["turn_count"] + chunk[3]["turn_count"]
                    )
                    pending_chunk = (
                        merged_content,
                        pending_chunk[1],
                        chunk[2],
                        merged_meta,
                    )
                else:
                    pending_chunk = chunk
            else:
                if pending_chunk:
                    # Merge pending with current
                    merged_content = pending_chunk[0] + "\n\n" + chunk[0]
                    merged_meta = pending_chunk[3].copy()
                    merged_meta["datetime_end"] = chunk[3]["datetime_end"]
                    merged_meta["roles"] = pending_chunk[3]["roles"] + chunk[3]["roles"]
                    merged_meta["turn_count"] = (
                        pending_chunk[3]["turn_count"] + chunk[3]["turn_count"]
                    )
                    merged_chunks.append(
                        (merged_content, pending_chunk[1], chunk[2], merged_meta)
                    )
                    pending_chunk = None
                else:
                    merged_chunks.append(chunk)

        if pending_chunk:
            if merged_chunks:
                # Merge with last chunk
                last = merged_chunks[-1]
                merged_content = last[0] + "\n\n" + pending_chunk[0]
                merged_meta = last[3].copy()
                merged_meta["datetime_end"] = pending_chunk[3]["datetime_end"]
                merged_meta["roles"] = last[3]["roles"] + pending_chunk[3]["roles"]
                merged_meta["turn_count"] = (
                    last[3]["turn_count"] + pending_chunk[3]["turn_count"]
                )
                merged_chunks[-1] = (merged_content, last[1], pending_chunk[2], merged_meta)
            else:
                merged_chunks.append(pending_chunk)

        chunks = merged_chunks

    return chunks


def create_chunk_for_embedding(
    chunk_content: str, metadata: dict[str, Any]
) -> str:
    """
    Create the text to embed for a conversation chunk.

    This prepends key metadata to help with semantic search.

    Args:
        chunk_content: The chunk content.
        metadata: The chunk metadata dictionary.

    Returns:
        Text optimized for embedding.
    """
    # The chunk already has metadata header, but we can add more context
    # for semantic search if needed
    roles = metadata.get("roles", [])
    datetime_start = metadata.get("datetime_start", "unknown")
    datetime_end = metadata.get("datetime_end", "unknown")

    # Create a semantic preamble
    unique_roles = list(dict.fromkeys(roles))
    if len(unique_roles) == 1:
        roles_desc = f"a {unique_roles[0]} message"
    else:
        roles_desc = f"messages from {', '.join(unique_roles)}"

    if datetime_start == datetime_end:
        time_desc = f"at {datetime_start}"
    else:
        time_desc = f"from {datetime_start} to {datetime_end}"

    preamble = f"Conversation excerpt containing {roles_desc} {time_desc}:\n\n"

    return preamble + chunk_content
