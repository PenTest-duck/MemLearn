"""
MemLearn Playground - Multi-Agent Persistent Memory Demo (LangChain Edition)

This playground demonstrates:
1. Multi-agent support with persistent memory
2. Session management and agent switching
3. High observability into MemLearn operations
4. Complex tasks that let agents discover memory tools organically
5. Multi-model support via LangChain (OpenAI, Anthropic, Google, etc.)

Usage:
    python playground/main.py           # Interactive menu
    python playground/main.py --demo    # Run automated demo

Model Configuration:
    Set MEMLEARN_MODEL env var or use 'model' command to switch:
    - "openai:gpt-5.2" (default)
    - "anthropic:claude-sonnet-4-20250514"
    - "google_genai:gemini-2.5-pro"
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlearn import MemFS, MemLearnConfig, get_memfs_system_prompt_with_note
from memlearn.tools import LangChainToolProvider
from memlearn.types import SessionStatus

load_dotenv()

# Configuration
# Model format: "provider:model" e.g. "openai:gpt-5.2", "anthropic:claude-sonnet-4-20250514", "google_genai:gemini-2.5-pro"
# Supported providers: openai, anthropic, google_genai, bedrock, azure_openai, etc.
MODEL_NAME = os.getenv("MEMLEARN_MODEL", "openai:gpt-5.2")
MEMLEARN_HOME = Path.home() / ".memlearn"


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


def print_header(text: str, color: str = Colors.CYAN):
    """Print a styled header."""
    width = 70
    print(f"\n{color}{'═' * width}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{color}{'═' * width}{Colors.RESET}")


def print_subheader(text: str):
    """Print a styled subheader."""
    print(f"\n{Colors.GRAY}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.WHITE}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.GRAY}{'─' * 50}{Colors.RESET}")


def print_info(label: str, value: str):
    """Print an info line."""
    print(f"  {Colors.GRAY}{label}:{Colors.RESET} {value}")


def print_success(msg: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_error(msg: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_warning(msg: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def get_model(model_string: str = MODEL_NAME):
    """
    Initialize a chat model from a model string.

    Format: "provider:model" e.g.:
    - "openai:gpt-5.2"
    - "anthropic:claude-sonnet-4-20250514"
    - "google_genai:gemini-2.5-pro"

    Or just "model" which defaults to OpenAI.
    """
    return init_chat_model(model_string, streaming=True)


def create_multiline_prompt_session() -> PromptSession:
    """Create a prompt session that supports multi-line input.

    Enter: Insert newline
    Meta+Enter (Alt+Enter) or Escape then Enter: Submit
    """
    bindings = KeyBindings()

    @bindings.add(Keys.Enter)
    def _(event):
        """Enter inserts a newline."""
        event.current_buffer.insert_text("\n")

    @bindings.add(Keys.Escape, Keys.Enter)
    def _(event):
        """Escape + Enter submits the input."""
        event.current_buffer.validate_and_handle()

    @bindings.add("c-d")
    def _(event):
        """Ctrl+D also submits (common Unix pattern)."""
        event.current_buffer.validate_and_handle()

    return PromptSession(key_bindings=bindings, multiline=True)


def get_system_prompt(agent_name: str, memory_note: str) -> str:
    """Build a system prompt with dynamic memory note injection.

    The memory note provides context about what's currently stored in memory,
    helping the agent understand available knowledge without exploring from scratch.
    """
    # Use the new function that injects the memory note into the prompt
    memfs_prompt = get_memfs_system_prompt_with_note(memory_note, extended=True)

    return f"""You are {agent_name}, an AI assistant with persistent memory.

You have access to a personal memory filesystem (MemFS) where you can store and retrieve information. This memory persists across conversations - anything you save will be there next time we talk.

{memfs_prompt}

## How to Work

You have complete freedom in how you use your memory system. Some agents prefer detailed notes, others prefer structured data, some use it sparingly. Find what works for you.

The key insight: your memory is YOUR tool. Use it when it helps you be more effective - for remembering context, tracking progress, storing insights, or building knowledge over time.

When checking your memory, remember to look in BOTH /memory/ (your notes) AND /raw/ (system data like conversation history). The "Current Memory State" section above tells you what's available.

Be natural, helpful, and thoughtful. Your memory is an extension of your capabilities."""


class MemLearnObserver:
    """Observability layer for MemLearn operations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.operation_log = []
        self.session_stats = {
            "tool_calls": 0,
            "files_created": 0,
            "files_read": 0,
            "files_edited": 0,
            "searches": 0,
            "errors": 0,
        }

    def log_tool_call(self, tool_name: str, args: dict, result: dict):
        """Log a tool call with full details."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        entry = {
            "timestamp": timestamp,
            "tool": tool_name,
            "args": args,
            "result": result,
        }
        self.operation_log.append(entry)

        # Update stats
        self.session_stats["tool_calls"] += 1

        tool_base = tool_name.replace("memfs_", "")
        if tool_base == "create":
            self.session_stats["files_created"] += 1
        elif tool_base == "read":
            self.session_stats["files_read"] += 1
        elif tool_base == "edit":
            self.session_stats["files_edited"] += 1
        elif tool_base == "search":
            self.session_stats["searches"] += 1

        if result.get("status") == "error":
            self.session_stats["errors"] += 1

        if self.verbose:
            self._display_operation(entry)

    def _display_operation(self, entry: dict):
        """Display a single operation with full details."""
        tool = entry["tool"].replace("memfs_", "")
        status = entry["result"].get("status", "unknown")

        # Color based on operation type
        tool_colors = {
            "read": Colors.BLUE,
            "create": Colors.GREEN,
            "edit": Colors.YELLOW,
            "delete": Colors.RED,
            "mkdir": Colors.GREEN,
            "list": Colors.CYAN,
            "move": Colors.MAGENTA,
            "search": Colors.MAGENTA,
            "peek": Colors.CYAN,
            "compact": Colors.YELLOW,
        }
        color = tool_colors.get(tool, Colors.WHITE)
        status_color = Colors.GREEN if status == "success" else Colors.RED
        status_icon = "✓" if status == "success" else "✗"

        # Build display
        print(f"\n{Colors.GRAY}┌─ {entry['timestamp']} {'─' * 43}┐{Colors.RESET}")
        print(
            f"{Colors.GRAY}│{Colors.RESET} {color}{Colors.BOLD}TOOL CALL: {entry['tool']}{Colors.RESET}"
        )
        print(f"{Colors.GRAY}│{Colors.RESET}")

        # Show ALL arguments
        args = entry["args"]
        print(f"{Colors.GRAY}│{Colors.RESET} {Colors.CYAN}Arguments:{Colors.RESET}")
        if args:
            for key, value in args.items():
                # Format value based on type and length
                if isinstance(value, str):
                    if len(value) > 200:
                        # Truncate long strings but show more than before
                        formatted_value = (
                            f'"{value[:200]}..." ({len(value)} chars total)'
                        )
                    elif "\n" in value:
                        # Multi-line content - show with indentation
                        lines = value.split("\n")
                        if len(lines) > 10:
                            preview_lines = lines[:10]
                            formatted_value = (
                                f'"""\n{Colors.GRAY}│{Colors.RESET}     '
                                + f"\n{Colors.GRAY}│{Colors.RESET}     ".join(
                                    preview_lines
                                )
                                + f'\n{Colors.GRAY}│{Colors.RESET}     ... ({len(lines)} lines total)\n{Colors.GRAY}│{Colors.RESET}   """'
                            )
                        else:
                            formatted_value = (
                                f'"""\n{Colors.GRAY}│{Colors.RESET}     '
                                + f"\n{Colors.GRAY}│{Colors.RESET}     ".join(lines)
                                + f'\n{Colors.GRAY}│{Colors.RESET}   """'
                            )
                    else:
                        formatted_value = f'"{value}"'
                elif isinstance(value, bool):
                    formatted_value = f"{Colors.MAGENTA}{value}{Colors.RESET}"
                elif isinstance(value, (int, float)):
                    formatted_value = f"{Colors.YELLOW}{value}{Colors.RESET}"
                else:
                    formatted_value = json.dumps(value, indent=2)
                    if "\n" in formatted_value:
                        formatted_value = formatted_value.replace(
                            "\n", f"\n{Colors.GRAY}│{Colors.RESET}     "
                        )
                print(
                    f"{Colors.GRAY}│{Colors.RESET}   {Colors.WHITE}{key}{Colors.RESET}: {formatted_value}"
                )
        else:
            print(
                f"{Colors.GRAY}│{Colors.RESET}   {Colors.DIM}(no arguments){Colors.RESET}"
            )

        print(f"{Colors.GRAY}│{Colors.RESET}")
        print(
            f"{Colors.GRAY}│{Colors.RESET} {status_color}{status_icon} Result:{Colors.RESET}"
        )

        # Show full result
        result = entry["result"]
        self._display_result_data(result)

        print(f"{Colors.GRAY}└{'─' * 58}┘{Colors.RESET}")

    def _display_result_data(self, result: dict, indent: int = 3):
        """Display result data with proper formatting."""
        indent_str = " " * indent

        for key, value in result.items():
            if key == "status":
                continue  # Already shown in header

            if isinstance(value, str):
                if len(value) > 300:
                    # Truncate very long strings
                    print(
                        f'{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: "{value[:300]}..." ({len(value)} chars)'
                    )
                elif "\n" in value:
                    lines = value.split("\n")
                    if len(lines) > 15:
                        print(
                            f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}:"
                        )
                        for line in lines[:15]:
                            print(
                                f"{Colors.GRAY}│{Colors.RESET}{indent_str}  {Colors.DIM}{line}{Colors.RESET}"
                            )
                        print(
                            f"{Colors.GRAY}│{Colors.RESET}{indent_str}  {Colors.DIM}... ({len(lines)} lines total){Colors.RESET}"
                        )
                    else:
                        print(
                            f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}:"
                        )
                        for line in lines:
                            print(
                                f"{Colors.GRAY}│{Colors.RESET}{indent_str}  {Colors.DIM}{line}{Colors.RESET}"
                            )
                else:
                    print(
                        f'{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: "{value}"'
                    )
            elif isinstance(value, bool):
                print(
                    f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: {Colors.MAGENTA}{value}{Colors.RESET}"
                )
            elif isinstance(value, (int, float)):
                print(
                    f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: {Colors.YELLOW}{value}{Colors.RESET}"
                )
            elif isinstance(value, list):
                print(
                    f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: ["
                )
                for i, item in enumerate(value[:10]):  # Limit to first 10 items
                    if isinstance(item, dict):
                        # Compact dict display for list items
                        item_str = json.dumps(item, separators=(",", ":"))
                        if len(item_str) > 80:
                            item_str = item_str[:80] + "..."
                        print(
                            f"{Colors.GRAY}│{Colors.RESET}{indent_str}    {Colors.DIM}{item_str}{Colors.RESET}"
                        )
                    else:
                        print(
                            f"{Colors.GRAY}│{Colors.RESET}{indent_str}    {Colors.DIM}{item}{Colors.RESET}"
                        )
                if len(value) > 10:
                    print(
                        f"{Colors.GRAY}│{Colors.RESET}{indent_str}    {Colors.DIM}... ({len(value)} items total){Colors.RESET}"
                    )
                print(f"{Colors.GRAY}│{Colors.RESET}{indent_str}]")
            elif isinstance(value, dict):
                print(
                    f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}:"
                )
                for k, v in list(value.items())[:10]:
                    v_str = str(v)
                    if len(v_str) > 60:
                        v_str = v_str[:60] + "..."
                    print(
                        f"{Colors.GRAY}│{Colors.RESET}{indent_str}    {Colors.DIM}{k}: {v_str}{Colors.RESET}"
                    )
                if len(value) > 10:
                    print(
                        f"{Colors.GRAY}│{Colors.RESET}{indent_str}    {Colors.DIM}... ({len(value)} keys total){Colors.RESET}"
                    )
            elif value is None:
                print(
                    f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: {Colors.DIM}null{Colors.RESET}"
                )
            else:
                print(
                    f"{Colors.GRAY}│{Colors.RESET}{indent_str}{Colors.WHITE}{key}{Colors.RESET}: {value}"
                )

    def display_stats(self):
        """Display session statistics."""
        print_subheader("Session Statistics")
        stats = self.session_stats
        print(f"  {Colors.CYAN}Tool Calls:{Colors.RESET} {stats['tool_calls']}")
        print(f"  {Colors.GREEN}Files Created:{Colors.RESET} {stats['files_created']}")
        print(f"  {Colors.BLUE}Files Read:{Colors.RESET} {stats['files_read']}")
        print(f"  {Colors.YELLOW}Files Edited:{Colors.RESET} {stats['files_edited']}")
        print(f"  {Colors.MAGENTA}Searches:{Colors.RESET} {stats['searches']}")
        if stats["errors"] > 0:
            print(f"  {Colors.RED}Errors:{Colors.RESET} {stats['errors']}")

    def display_recent_operations(self, count: int = 5):
        """Display recent operations."""
        if not self.operation_log:
            print(f"  {Colors.DIM}No operations yet{Colors.RESET}")
            return

        recent = self.operation_log[-count:]
        for entry in recent:
            tool = entry["tool"].replace("memfs_", "")
            status = "✓" if entry["result"].get("status") == "success" else "✗"
            path = entry["args"].get("path", entry["args"].get("query", ""))[:30]
            print(
                f"  {Colors.GRAY}{entry['timestamp']}{Colors.RESET} [{tool}] {status} {path}"
            )


class PlaygroundSession:
    """Manages a single agent session."""

    def __init__(
        self,
        agent_name: str,
        observer: MemLearnObserver,
        model_string: str = MODEL_NAME,
    ):
        self.agent_name = agent_name
        self.observer = observer
        self.model_string = model_string
        self.memfs = None
        self.tool_provider = None
        self.model = None
        self.messages = []  # LangChain message objects
        self.turn_count = 0

    def start(self):
        """Start or resume the session."""
        print(
            f"\n{Colors.CYAN}Starting session for {Colors.BOLD}{self.agent_name}{Colors.RESET}{Colors.CYAN}...{Colors.RESET}"
        )
        print(f"  {Colors.GRAY}Model: {self.model_string}{Colors.RESET}")

        # Use persistent config with debug enabled
        config = MemLearnConfig.default_persistent()
        config.debug = True

        # Create MemFS with persistence (read_only=False for full read-write access)
        self.memfs = MemFS.for_agent(self.agent_name, config, read_only=False)
        self.tool_provider = self.memfs.get_tool_provider(
            enable_bash=True  # WARN: security issue - allows arbitrary shell commands
        )

        # Initialize the LangChain model with tools bound
        self.model = get_model(self.model_string)
        tools = self.tool_provider.get_tools()
        self.model = self.model.bind_tools(tools)

        # Get the dynamic memory note for system prompt injection
        memory_note = self.memfs.get_memory_note()

        # Initialize messages with system prompt (includes memory note)
        system_prompt = get_system_prompt(self.agent_name, memory_note)
        self.messages = [SystemMessage(content=system_prompt)]

        print_success(f"Session started")
        print_info("Agent ID", self.memfs.config.agent_id or "N/A")
        print_info("Session ID", self.memfs.config.session_id or "N/A")
        print_info("Sandbox", self.memfs.sandbox.root_path)

        # Show the memory note and status
        self._show_memory_status(memory_note)

    def _show_memory_status(self, memory_note: str | None = None):
        """Display current memory filesystem status including the dynamic memory note."""
        print_subheader("Memory Status")

        # Display the dynamic memory note
        if memory_note:
            print(
                f"  {Colors.CYAN}{Colors.BOLD}Memory Note (injected into system prompt):{Colors.RESET}"
            )
            # Indent and wrap the note nicely
            for line in memory_note.split("\n"):
                print(f"  {Colors.WHITE}{line}{Colors.RESET}")
            print()

        try:
            # List /memory contents
            result = self.memfs.list_directory("/memory", max_depth=2)
            if result.status == "success" and result.data:
                entries = result.data.get("entries", [])
                if entries:
                    print(
                        f"  {Colors.GREEN}Found {len(entries)} items in /memory:{Colors.RESET}"
                    )
                    for entry in entries[:10]:
                        icon = "📁" if entry.get("is_dir") else "📄"
                        print(f"    {icon} {entry.get('name', 'unknown')}")
                    if len(entries) > 10:
                        print(
                            f"    {Colors.DIM}... and {len(entries) - 10} more{Colors.RESET}"
                        )
                else:
                    print(
                        f"  {Colors.DIM}/memory is empty - this appears to be a fresh agent{Colors.RESET}"
                    )
        except Exception:
            print(f"  {Colors.DIM}Unable to read memory status{Colors.RESET}")

    def end(self, status: SessionStatus = SessionStatus.COMPLETED):
        """End the session gracefully."""
        if self.memfs:
            print(f"\n{Colors.CYAN}Ending session...{Colors.RESET}")
            self.observer.display_stats()
            self.memfs.spindown(status=status)
            print_success("Session ended and memory persisted")
            self.memfs = None

    def chat_turn(self, user_input: str, max_tool_rounds: int = 15):
        """Execute a single chat turn using LangChain streaming."""
        self.turn_count += 1
        self.messages.append(HumanMessage(content=user_input))

        # Record in conversation history
        if self.memfs:
            self.memfs.append_conversation_message(
                {"role": "user", "content": user_input}
            )

        for round_num in range(max_tool_rounds):
            ai_message, usage_info = self._stream_response()

            # Add the AI message to our messages
            self.messages.append(ai_message)

            # Record in conversation history
            if self.memfs and ai_message.content:
                self.memfs.append_conversation_message(
                    {"role": "assistant", "content": ai_message.content}
                )

            # Check if we have tool calls
            if not ai_message.tool_calls:
                self._display_usage(usage_info)
                break

            # Execute tool calls
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                arguments = tool_call["args"]

                # Execute and observe
                result_str = self.tool_provider.execute_tool(tool_name, arguments)
                try:
                    result_data = json.loads(result_str)
                except json.JSONDecodeError:
                    result_data = {"status": "unknown", "message": result_str[:100]}

                # Log to observer
                self.observer.log_tool_call(tool_name, arguments, result_data)

                # Add tool result to messages
                self.messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tool_call["id"],
                    )
                )

    def _stream_response(self) -> tuple[AIMessage, dict | None]:
        """Stream response from the LangChain model."""
        full_content = ""
        tool_calls = []
        tool_call_chunks = {}  # Track tool call chunks by index
        usage_info = None

        print(
            f"\n{Colors.BLUE}{Colors.BOLD}{self.agent_name}:{Colors.RESET} ",
            end="",
            flush=True,
        )

        for chunk in self.model.stream(self.messages):
            # Handle content streaming
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_content += chunk.content

            # Handle tool call chunks - they arrive progressively
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc_chunk in chunk.tool_call_chunks:
                    idx = tc_chunk.get("index", 0)

                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": "",
                            "name": "",
                            "args": "",
                        }

                    if tc_chunk.get("id"):
                        tool_call_chunks[idx]["id"] = tc_chunk["id"]
                    if tc_chunk.get("name"):
                        tool_call_chunks[idx]["name"] = tc_chunk["name"]
                    if tc_chunk.get("args"):
                        tool_call_chunks[idx]["args"] += tc_chunk["args"]

            # Try to extract usage info from response metadata
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                meta = chunk.response_metadata
                if "usage" in meta:
                    usage = meta["usage"]
                    usage_info = {
                        "prompt_tokens": usage.get(
                            "prompt_tokens", usage.get("input_tokens", 0)
                        ),
                        "completion_tokens": usage.get(
                            "completion_tokens", usage.get("output_tokens", 0)
                        ),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                # OpenAI-style token usage
                if "token_usage" in meta:
                    usage = meta["token_usage"]
                    usage_info = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

        print()

        # Build final tool calls list from accumulated chunks
        for idx in sorted(tool_call_chunks.keys()):
            tc = tool_call_chunks[idx]
            if tc["id"] and tc["name"]:
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": args,
                    }
                )

        # Create AIMessage with content and tool calls
        ai_message = AIMessage(content=full_content, tool_calls=tool_calls)
        return ai_message, usage_info

    def _display_usage(self, usage_info: dict | None):
        """Display token usage."""
        if not usage_info:
            print(f"{Colors.DIM}[Turn {self.turn_count}]{Colors.RESET}")
            return
        print(
            f"{Colors.DIM}[Turn {self.turn_count} | Tokens: {usage_info.get('total_tokens', '?')} (in: {usage_info.get('prompt_tokens', '?')}, out: {usage_info.get('completion_tokens', '?')})]{Colors.RESET}"
        )


def list_existing_agents() -> list[dict]:
    """List all existing agents from the database."""
    agents = []

    # Check if memlearn database exists
    db_path = MEMLEARN_HOME / "memlearn.db"
    if not db_path.exists():
        return agents

    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT agent_id, name, created_at FROM agents ORDER BY created_at DESC"
        )
        for row in cursor.fetchall():
            # Get session count
            cursor.execute(
                "SELECT COUNT(*) FROM sessions WHERE agent_id = ?", (row["agent_id"],)
            )
            session_count = cursor.fetchone()[0]

            agents.append(
                {
                    "agent_id": row["agent_id"],
                    "name": row["name"],
                    "created_at": datetime.fromtimestamp(row["created_at"]).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "sessions": session_count,
                }
            )

        conn.close()
    except Exception as e:
        print_warning(f"Could not read agents: {e}")

    return agents


def display_agents_table(agents: list[dict]):
    """Display agents in a nice table format."""
    if not agents:
        print(
            f"\n  {Colors.DIM}No agents found. Create your first agent!{Colors.RESET}"
        )
        return

    print(
        f"\n  {Colors.BOLD}{'#':<4} {'Name':<20} {'Sessions':<10} {'Created':<20}{Colors.RESET}"
    )
    print(f"  {Colors.GRAY}{'─' * 54}{Colors.RESET}")

    for i, agent in enumerate(agents, 1):
        print(
            f"  {Colors.CYAN}{i:<4}{Colors.RESET} {agent['name']:<20} {agent['sessions']:<10} {agent['created_at']:<20}"
        )


def get_complex_demo_tasks() -> list[str]:
    """
    Return complex tasks that encourage organic memory usage.
    These tasks are challenging but don't prescribe specific tools.
    """
    return [
        # Task 1: Open-ended research and synthesis
        """I'm building a startup that helps developers be more productive. 
        I need you to think deeply about what makes developers productive vs unproductive, 
        identify the key bottlenecks, and develop a framework for thinking about this problem.
        Take your time and be thorough - I want your genuine insights, not surface-level observations.""",
        # Task 2: Building on previous work (tests memory)
        """Based on your previous analysis, what would be the top 3 product ideas that could 
        address the biggest productivity bottlenecks you identified? For each idea, explain 
        why it addresses the problem and what would make it succeed or fail.""",
        # Task 3: Reflection and meta-learning
        """I'm curious - as you've been thinking through this problem, have you noticed any 
        patterns in how you approach complex analysis? What strategies seem to work well for you? 
        I'm interested in understanding your thinking process.""",
        # Task 4: Application with constraints
        """Let's get practical. If I only had $50k and 3 months to build an MVP, which of your 
        product ideas would you recommend and why? Walk me through a rough plan for what we'd 
        build first and how we'd validate it.""",
        # Task 5: New session simulation (for multi-session testing)
        """Hey! I was talking to you earlier about the developer productivity startup. 
        Can you remind me what we discussed and where we left off? I want to continue 
        our conversation.""",
    ]


def run_demo(observer: MemLearnObserver):
    """Run the automated demo showcasing persistence."""
    print_header("MemLearn Persistence Demo", Colors.MAGENTA)
    print(
        f"""
{Colors.WHITE}This demo shows how agents maintain memory across sessions.{Colors.RESET}

{Colors.DIM}We'll run through a multi-turn conversation with an agent, then 
start a NEW session with the SAME agent to see if it remembers.{Colors.RESET}
"""
    )

    agent_name = "demo-researcher"
    tasks = get_complex_demo_tasks()

    # First session
    print_header(f"Session 1 with '{agent_name}'", Colors.GREEN)
    session = PlaygroundSession(agent_name, observer, model_string=MODEL_NAME)
    session.start()

    try:
        for i, task in enumerate(tasks[:4], 1):
            print(f"\n{Colors.YELLOW}{'═' * 60}{Colors.RESET}")
            print(f"{Colors.YELLOW}Turn {i}/4{Colors.RESET}")
            print(f"{Colors.YELLOW}{'═' * 60}{Colors.RESET}")
            print(
                f"\n{Colors.GREEN}You:{Colors.RESET} {task[:100]}..."
                if len(task) > 100
                else f"\n{Colors.GREEN}You:{Colors.RESET} {task}"
            )

            session.chat_turn(task)

            if i < 4:
                input(f"\n{Colors.DIM}[Press Enter for next turn...]{Colors.RESET}")
    finally:
        session.end()

    # Pause between sessions
    print_header("Session 1 Complete", Colors.CYAN)
    print(
        f"""
{Colors.WHITE}The agent's memory has been persisted to disk.{Colors.RESET}
{Colors.DIM}Location: {MEMLEARN_HOME}/persistent/agents/{Colors.RESET}

{Colors.YELLOW}Now we'll start a NEW session with the SAME agent.{Colors.RESET}
{Colors.YELLOW}Watch to see if it remembers the previous conversation!{Colors.RESET}
"""
    )
    input(f"{Colors.DIM}[Press Enter to start Session 2...]{Colors.RESET}")

    # Second session - test memory persistence
    print_header(f"Session 2 with '{agent_name}' (Testing Memory)", Colors.GREEN)

    observer2 = MemLearnObserver(verbose=True)
    session2 = PlaygroundSession(agent_name, observer2, model_string=MODEL_NAME)
    session2.start()

    try:
        # Use the last task which asks about previous conversation
        print(f"\n{Colors.YELLOW}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.YELLOW}Testing Memory Recall{Colors.RESET}")
        print(f"{Colors.YELLOW}{'═' * 60}{Colors.RESET}")
        print(f"\n{Colors.GREEN}You:{Colors.RESET} {tasks[4]}")

        session2.chat_turn(tasks[4])
    finally:
        session2.end()

    # Final summary
    print_header("Demo Complete!", Colors.MAGENTA)
    print(
        f"""
{Colors.GREEN}What we demonstrated:{Colors.RESET}
  1. Created an agent with persistent memory
  2. Had a multi-turn conversation
  3. Agent stored insights in its memory filesystem
  4. Started a NEW session (simulating a new conversation)
  5. Agent recalled information from its persistent memory

{Colors.CYAN}The memory persists in: {MEMLEARN_HOME}/persistent/{Colors.RESET}
"""
    )


def interactive_menu():
    """Main interactive menu."""
    global MODEL_NAME
    observer = MemLearnObserver(verbose=True)
    current_session = None

    while True:
        print_header("MemLearn Playground", Colors.CYAN)
        print(f"  {Colors.GRAY}Model: {MODEL_NAME}{Colors.RESET}")

        # Show existing agents
        agents = list_existing_agents()
        print_subheader("Existing Agents")
        display_agents_table(agents)

        # Menu options
        print(
            f"""
{Colors.WHITE}Commands:{Colors.RESET}
  {Colors.CYAN}new <name>{Colors.RESET}     - Create and start a new agent
  {Colors.CYAN}start <name>{Colors.RESET}   - Start session with existing agent  
  {Colors.CYAN}start <#>{Colors.RESET}      - Start session with agent by number
  {Colors.CYAN}model <model>{Colors.RESET}  - Switch model (e.g. anthropic:claude-sonnet-4-20250514)
  {Colors.CYAN}demo{Colors.RESET}           - Run the automated persistence demo
  {Colors.CYAN}stats{Colors.RESET}          - Show current session statistics
  {Colors.CYAN}inspect{Colors.RESET}        - Inspect current agent's memory
  {Colors.CYAN}quit{Colors.RESET}           - Exit playground
"""
        )

        try:
            cmd = input(f"{Colors.GREEN}playground>{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if action == "quit" or action == "exit":
            if current_session:
                current_session.end()
            print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}")
            break

        elif action == "demo":
            if current_session:
                current_session.end()
                current_session = None
            run_demo(observer)

        elif action == "new":
            if not arg:
                print_error("Please provide a name: new <agent-name>")
                continue

            if current_session:
                current_session.end()

            observer = MemLearnObserver(verbose=True)
            current_session = PlaygroundSession(arg, observer, model_string=MODEL_NAME)
            current_session.start()

            # Enter chat mode
            run_chat_mode(current_session, observer)
            current_session = None

        elif action == "start":
            if not arg:
                print_error("Please provide a name or number: start <agent-name|#>")
                continue

            # Check if it's a number
            agent_name = arg
            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(agents):
                    agent_name = agents[idx]["name"]
                else:
                    print_error(f"Invalid agent number. Choose 1-{len(agents)}")
                    continue

            if current_session:
                current_session.end()

            observer = MemLearnObserver(verbose=True)
            current_session = PlaygroundSession(
                agent_name, observer, model_string=MODEL_NAME
            )
            current_session.start()

            # Enter chat mode
            run_chat_mode(current_session, observer)
            current_session = None

        elif action == "model":
            if not arg:
                print(f"Current model: {Colors.CYAN}{MODEL_NAME}{Colors.RESET}")
                print(f"\n{Colors.WHITE}Available model formats:{Colors.RESET}")
                print(f"  {Colors.GRAY}openai:gpt-5.2{Colors.RESET}")
                print(f"  {Colors.GRAY}openai:gpt-4.1{Colors.RESET}")
                print(
                    f"  {Colors.GRAY}anthropic:claude-sonnet-4-20250514{Colors.RESET}"
                )
                print(
                    f"  {Colors.GRAY}anthropic:claude-3-5-sonnet-latest{Colors.RESET}"
                )
                print(f"  {Colors.GRAY}google_genai:gemini-2.5-pro{Colors.RESET}")
                print(f"  {Colors.GRAY}google_genai:gemini-2.0-flash{Colors.RESET}")
                continue

            # Check API key for the new model
            if not check_api_keys(arg):
                continue

            MODEL_NAME = arg
            print_success(f"Model switched to: {MODEL_NAME}")

        elif action == "stats":
            observer.display_stats()
            observer.display_recent_operations()

        elif action == "inspect":
            if not current_session:
                print_warning("No active session. Start a session first.")
                continue
            inspect_memory(current_session)

        else:
            print_warning(f"Unknown command: {action}")


def run_chat_mode(session: PlaygroundSession, observer: MemLearnObserver):
    """Run interactive chat mode with an agent."""
    multiline_mode = False
    prompt_session = create_multiline_prompt_session()

    def print_mode_hint():
        if multiline_mode:
            print(
                f"{Colors.DIM}Multi-line mode: Enter for new line, Escape+Enter or Ctrl+D to send{Colors.RESET}"
            )
        else:
            print(f"{Colors.DIM}Single-line mode: Enter to send{Colors.RESET}")

    print(
        f"""
{Colors.DIM}{'─' * 60}
Chat mode with {session.agent_name}
Commands: /stats /inspect /memory /multiline /end /help
{'─' * 60}{Colors.RESET}
"""
    )
    print_mode_hint()

    try:
        while True:
            try:
                if multiline_mode:
                    user_input = prompt_session.prompt(
                        f"\n{Colors.GREEN}You:{Colors.RESET} "
                    ).strip()
                else:
                    user_input = input(f"\n{Colors.GREEN}You:{Colors.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input[1:].lower()

                if cmd == "end" or cmd == "quit":
                    break
                elif cmd == "stats":
                    observer.display_stats()
                elif cmd == "inspect" or cmd == "memory":
                    inspect_memory(session)
                elif cmd == "multiline":
                    multiline_mode = not multiline_mode
                    print_mode_hint()
                elif cmd == "help":
                    print(
                        f"""
{Colors.CYAN}Available commands:{Colors.RESET}
  /stats     - Show session statistics
  /inspect   - Inspect memory filesystem
  /memory    - Alias for /inspect
  /multiline - Toggle multi-line input mode
  /end       - End this session
  /help      - Show this help
"""
                    )
                else:
                    print_warning(f"Unknown command: /{cmd}")
                continue

            # Regular chat
            session.chat_turn(user_input)

    finally:
        session.end()


def inspect_memory(session: PlaygroundSession):
    """Inspect the current agent's memory filesystem."""
    if not session.memfs:
        print_warning("No active MemFS")
        return

    print_subheader("Memory Filesystem Inspector")

    # List all directories
    for dir_path in ["/memory", "/raw", "/tmp", "/mnt"]:
        result = session.memfs.list_directory(dir_path, max_depth=2)
        if result.status == "success" and result.data:
            entries = result.data.get("entries", [])
            print(f"\n  {Colors.CYAN}{dir_path}/{Colors.RESET} ({len(entries)} items)")
            for entry in entries[:5]:
                icon = "📁" if entry.get("is_dir") else "📄"
                size = entry.get("size", 0)
                size_str = f" ({size} bytes)" if size else ""
                print(f"    {icon} {entry.get('name', '?')}{size_str}")
            if len(entries) > 5:
                print(f"    {Colors.DIM}... and {len(entries) - 5} more{Colors.RESET}")


def check_api_keys(model_string: str) -> bool:
    """Check if the required API key is set for the given model provider."""
    provider = model_string.split(":")[0].lower() if ":" in model_string else "openai"

    key_mapping = {
        "openai": ("OPENAI_API_KEY", "OpenAI"),
        "anthropic": ("ANTHROPIC_API_KEY", "Anthropic"),
        "google_genai": ("GOOGLE_API_KEY", "Google (Gemini)"),
        "google": ("GOOGLE_API_KEY", "Google (Gemini)"),
        "bedrock": ("AWS_ACCESS_KEY_ID", "AWS Bedrock"),
        "azure_openai": ("AZURE_OPENAI_API_KEY", "Azure OpenAI"),
    }

    if provider in key_mapping:
        env_var, provider_name = key_mapping[provider]
        if not os.getenv(env_var):
            print_warning(f"{env_var} not set for {provider_name} models.")
            print(
                f"  Set it in your .env file or environment to use {provider_name} models."
            )
            return False
    return True


def main():
    """Main entry point."""
    # Check for at least one API key
    has_any_key = any(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
        ]
    )

    if not has_any_key:
        print_error("No API keys found.")
        print("Please set at least one of the following in your .env file:")
        print("  - OPENAI_API_KEY (for OpenAI models like gpt-5.2)")
        print("  - ANTHROPIC_API_KEY (for Anthropic models like claude-sonnet-4)")
        print("  - GOOGLE_API_KEY (for Google models like gemini-2.5-pro)")
        sys.exit(1)

    # Check if the default model's API key is available
    if not check_api_keys(MODEL_NAME):
        print_warning(
            f"Default model '{MODEL_NAME}' may not work without the required API key."
        )

    # Ensure memlearn home exists
    MEMLEARN_HOME.mkdir(parents=True, exist_ok=True)

    # Check command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            observer = MemLearnObserver(verbose=True)
            run_demo(observer)
            return
        elif sys.argv[1] == "--help":
            print(__doc__)
            print(f"\nCurrent model: {MODEL_NAME}")
            print("Set MEMLEARN_MODEL env var to change, e.g.:")
            print("  export MEMLEARN_MODEL='anthropic:claude-sonnet-4-20250514'")
            print("  export MEMLEARN_MODEL='google_genai:gemini-2.5-pro'")
            return

    # Run interactive menu
    interactive_menu()


if __name__ == "__main__":
    main()
