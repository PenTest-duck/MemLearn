"""
Prompt templates for MemLearn agent integration.

This module provides system prompt components that developers can include
in their agent's system prompt to help the agent understand and effectively
use MemFS for memory management.
"""

from __future__ import annotations

# Concise system prompt component for MemFS usage
MEMFS_SYSTEM_PROMPT = """## Memory System (MemFS)

You have access to MemFS, a persistent filesystem for storing and retrieving memory.

### Directory Structure
- `/memory/` - Long-term memory (persists across sessions) - use this for insights, notes, learnings
- `/tmp/` - Working memory (cleared after session) - use for drafts, scratchpad
- `/raw/` - Read-only reference materials
- `/mnt/` - Shared/mounted folders

### Available Tools
- `memfs_read` - Read file contents
- `memfs_create` - Create new files (provide description for searchability)
- `memfs_edit` - Edit files via string replacement
- `memfs_delete` - Delete files/folders
- `memfs_mkdir` - Create directories
- `memfs_list` - List directory contents
- `memfs_move` - Move/rename files
- `memfs_search` - Semantic + keyword search
- `memfs_peek` - View file metadata without reading

### Best Practices
1. Search before creating - check if relevant notes exist
2. Organize with folders in `/memory/`
3. Write detailed descriptions for searchability
4. Use `/tmp/` for working drafts, `/memory/` for permanent insights"""


# Extended system prompt with more detailed guidance
MEMFS_SYSTEM_PROMPT_EXTENDED = """## Memory System (MemFS)

You have access to MemFS - a persistent filesystem-based memory system that helps you store, organize, and retrieve knowledge across conversations.

### Directory Structure

- **`/memory/`** - Your main long-term memory folder
  - Persists across sessions
  - Full read/write access
  - Use for: insights, learnings, user preferences, notes, structured knowledge
  - Organize with subfolders (e.g., `/memory/topics/`, `/memory/projects/`)

- **`/tmp/`** - Temporary working memory
  - Cleared at end of session
  - Use for: scratchpad, drafts, working notes, temporary plans

- **`/raw/`** - Read-only reference materials
  - Contains: conversation history, uploaded artifacts, skill definitions
  - Cannot be modified by you

- **`/mnt/`** - Mounted shared folders
  - May contain shared memory from other agents/users
  - Use with awareness that others may read/write

### Available Tools

| Tool | Purpose |
|------|---------|
| `memfs_read` | Read file contents with line numbers |
| `memfs_create` | Create new files with content, description, and tags |
| `memfs_edit` | Edit files using exact string replacement |
| `memfs_delete` | Delete files or directories |
| `memfs_mkdir` | Create directories |
| `memfs_list` | List directory contents with metadata |
| `memfs_move` | Move or rename files/directories |
| `memfs_search` | Search by meaning (semantic) or keywords |
| `memfs_peek` | View metadata without reading full content |

### Best Practices

1. **Search before creating** - Use `memfs_search` to find existing relevant notes
2. **Organize logically** - Create topic folders in `/memory/`
3. **Write descriptions** - Good descriptions make files searchable
4. **Use tags** - Add relevant tags for filtering
5. **Keep notes detailed** - Include context and connections
6. **Use `/tmp/` for drafts** - Move to `/memory/` when finalized
7. **Review periodically** - Consolidate and organize knowledge"""


def get_memfs_system_prompt(extended: bool = False) -> str:
    """
    Get the MemFS system prompt component.

    Args:
        extended: If True, return the extended version with more detail.

    Returns:
        System prompt string to include in your agent's system prompt.

    Example:
        ```python
        from memlearn.prompts import get_memfs_system_prompt

        system_prompt = f'''You are a helpful assistant.

        {get_memfs_system_prompt()}

        Be helpful and use your memory effectively.
        '''
        ```
    """
    return MEMFS_SYSTEM_PROMPT_EXTENDED if extended else MEMFS_SYSTEM_PROMPT


# =============================================================================
# Conversation Summarization Prompts
# =============================================================================

CONVERSATION_SUMMARY_SYSTEM_PROMPT = """You are an expert at summarizing agent conversations. Create a concise but comprehensive summary that captures:

1. **Primary Task/Goal**: What was the main objective of this conversation?
2. **Key Actions Taken**: What significant actions or decisions were made?
3. **Outcomes**: What was accomplished? Were there any errors or issues?
4. **Important Details**: Any specific files, data, or information that was worked with.
5. **Learnings**: Any insights or patterns that might be useful for future sessions.

Be concise but don't omit important details. The summary should be useful for understanding what happened in this session without reading the full conversation."""


def get_conversation_summary_prompt(
    conversation_text: str,
    agent_name: str | None = None,
    session_context: str | None = None,
) -> tuple[str, str]:
    """
    Get the system and user prompts for conversation summarization.

    Args:
        conversation_text: The formatted conversation text to summarize.
        agent_name: Optional name of the agent.
        session_context: Optional additional context about the session.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = CONVERSATION_SUMMARY_SYSTEM_PROMPT

    if agent_name:
        system_prompt += f"\n\nThe conversation involves an agent named '{agent_name}'."

    if session_context:
        system_prompt += f"\n\nAdditional context: {session_context}"

    user_prompt = f"Please summarize this conversation:\n\n{conversation_text}"

    return system_prompt, user_prompt


# =============================================================================
# Memory Reflection Prompts (Future Use)
# =============================================================================

MEMORY_REFLECTION_SYSTEM_PROMPT = """You are an expert at analyzing and consolidating knowledge from an agent's memory system. Your task is to:

1. Identify patterns and recurring themes across memories
2. Consolidate redundant or overlapping information
3. Extract actionable insights and learnings
4. Suggest organizational improvements

Be thoughtful and preserve important nuances while eliminating redundancy."""


# =============================================================================
# Custom MemFS Prompt Builder
# =============================================================================


def get_custom_memfs_prompt(
    include_tools: bool = True,
    include_structure: bool = True,
    include_best_practices: bool = True,
    tool_prefix: str = "memfs",
) -> str:
    """
    Build a custom MemFS system prompt with selected sections.

    Args:
        include_tools: Include the tools reference section.
        include_structure: Include the directory structure section.
        include_best_practices: Include the best practices section.
        tool_prefix: The prefix used for tool names.

    Returns:
        Customized system prompt string.
    """
    sections = ["## Memory System (MemFS)\n"]
    sections.append(
        "You have access to MemFS, a persistent filesystem for storing and retrieving memory.\n"
    )

    if include_structure:
        sections.append("""
### Directory Structure
- `/memory/` - Long-term memory (persists across sessions)
- `/tmp/` - Working memory (cleared after session)
- `/raw/` - Read-only reference materials
- `/mnt/` - Shared/mounted folders
""")

    if include_tools:
        sections.append(f"""
### Available Tools
- `{tool_prefix}_read` - Read file contents
- `{tool_prefix}_create` - Create new files
- `{tool_prefix}_edit` - Edit files via string replacement
- `{tool_prefix}_delete` - Delete files/folders
- `{tool_prefix}_mkdir` - Create directories
- `{tool_prefix}_list` - List directory contents
- `{tool_prefix}_move` - Move/rename files
- `{tool_prefix}_search` - Semantic + keyword search
- `{tool_prefix}_peek` - View file metadata
""")

    if include_best_practices:
        sections.append("""
### Best Practices
1. Search before creating to find existing notes
2. Organize with folders in `/memory/`
3. Write detailed descriptions for searchability
4. Use `/tmp/` for drafts, `/memory/` for permanent insights
""")

    return "".join(sections)
