# MemLearn

An open-source agentic memory & continual learning system.

## Installation

1. Clone this repository and navigate to it
2. `python3.13 -m venv .venv` (at the moment, Python 3.14 doesn't work with
   ChromaDB)
3. `. .venv/bin/activate`
4. `uv sync --all-packages`

## Usage

Try it out locally with `uv run playground/main.py`.

You need to provide `OPENAI_API_KEY` and `COHERE_API_KEY` as environment
variables.

## Architecture: Context Window as Memory

We conceptualize the LLM's context window as a form of short-term memory. This
context window can be further divided into two parts:

1. **Core Context** - System prompt and tool schemas. This is the most precious
   token space as it's always present and sets the foundation for agent behavior.
2. **Rolling Context** - Conversation turns that accumulate during a session.

The closer to the "core" you get, the more valuable each token becomes.

### Dynamic Memory Notes

To help agents understand what's stored in their persistent memory without
consuming excessive context, MemLearn provides **dynamic memory notes** - a
concise summary (configurable, default max 1000 chars) that describes what's
currently in the MemFS.

This is a separation of:
- **Form**: The foundational structural architecture of memory (`/memory/`,
  `/raw/`, `/tmp/`, etc.)
- **Dynamics**: How the memory is actually being used and organized

#### How It Works

1. When a new MemFS volume is created, the memory note starts as a short sentence
   indicating empty memory.

2. Upon `spindown()`, MemFS runs an LLM to:
   - Review the existing memory note
   - List the entire filesystem structure
   - Generate an updated note reflecting the current state

3. On the next `spinup()`, the developer can retrieve this note and inject it
   into their agent's system prompt.

#### Usage Example

```python
from memlearn import MemFS, get_memfs_system_prompt_with_note

with MemFS.for_agent("my-agent") as memfs:
    # Get the dynamic memory note
    memory_note = memfs.get_memory_note()
    
    # Build system prompt with the note injected
    system_prompt = f"""You are a helpful assistant.

{get_memfs_system_prompt_with_note(memory_note)}

Be helpful and use your memory effectively.
"""
    
    # Use system_prompt with your LLM...
    tools = get_openai_tools(memfs)
    # ... run agent loop ...
```

#### Configuration

```python
from memlearn import MemLearnConfig

config = MemLearnConfig.default_persistent()

# Enable/disable memory note updates on spindown (default: True)
config.llm.update_memory_note_on_spindown = True

# Maximum tokens for the memory note (default: 256)
config.llm.memory_note_max_tokens = 256
```

#### Future Enhancements

- **Diff-based updates**: Using git-like version control to track changes
  between sessions and generate incremental note updates
- **Semantic importance ranking**: Prioritizing what to include in the note
  based on recency, frequency of access, and semantic relevance
