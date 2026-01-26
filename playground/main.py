"""
MemLearn Playground - Multi-Agent Persistent Memory Demo

This playground demonstrates:
1. Multi-agent support with persistent memory
2. Session management and agent switching
3. High observability into MemLearn operations
4. Complex tasks that let agents discover memory tools organically

Usage:
    python playground/main.py           # Interactive menu
    python playground/main.py --demo    # Run automated demo
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlearn import MemFS, MemLearnConfig, get_memfs_system_prompt
from memlearn.tools import OpenAIToolProvider
from memlearn.types import SessionStatus

load_dotenv()

# Configuration
MODEL_NAME = "gpt-5.2"  # SOTA
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


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_system_prompt(agent_name: str) -> str:
    """Build a system prompt that's less prescriptive about memory tools."""
    memfs_prompt = get_memfs_system_prompt(extended=True)

    return f"""You are {agent_name}, an AI assistant with persistent memory.

You have access to a personal memory filesystem (MemFS) where you can store and retrieve information. This memory persists across conversations - anything you save will be there next time we talk.

{memfs_prompt}

## How to Work

You have complete freedom in how you use your memory system. Some agents prefer detailed notes, others prefer structured data, some use it sparingly. Find what works for you.

The key insight: your memory is YOUR tool. Use it when it helps you be more effective - for remembering context, tracking progress, storing insights, or building knowledge over time.

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
        """Display a single operation with styling."""
        tool = entry["tool"].replace("memfs_", "")
        status = entry["result"].get("status", "unknown")
        message = entry["result"].get("message", "")

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

        # Build display
        print(
            f"\n{Colors.GRAY}┌─ {entry['timestamp']} ─────────────────────────────────────┐{Colors.RESET}"
        )
        print(
            f"{Colors.GRAY}│{Colors.RESET} {color}{Colors.BOLD}[{tool.upper()}]{Colors.RESET}"
        )

        # Show key arguments
        args = entry["args"]
        if "path" in args:
            print(
                f"{Colors.GRAY}│{Colors.RESET}   Path: {Colors.WHITE}{args['path']}{Colors.RESET}"
            )
        if "query" in args:
            print(
                f"{Colors.GRAY}│{Colors.RESET}   Query: {Colors.WHITE}{args['query']}{Colors.RESET}"
            )
        if "content" in args:
            content_preview = (
                args["content"][:50] + "..."
                if len(args.get("content", "")) > 50
                else args.get("content", "")
            )
            print(
                f"{Colors.GRAY}│{Colors.RESET}   Content: {Colors.DIM}{content_preview}{Colors.RESET}"
            )

        # Show result
        if status == "success":
            print(
                f"{Colors.GRAY}│{Colors.RESET}   {Colors.GREEN}✓ {message}{Colors.RESET}"
            )
        else:
            print(
                f"{Colors.GRAY}│{Colors.RESET}   {Colors.RED}✗ {message}{Colors.RESET}"
            )

        print(
            f"{Colors.GRAY}└──────────────────────────────────────────────────────┘{Colors.RESET}"
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

    def __init__(self, agent_name: str, observer: MemLearnObserver):
        self.agent_name = agent_name
        self.observer = observer
        self.memfs = None
        self.tool_provider = None
        self.messages = []
        self.turn_count = 0

    def start(self):
        """Start or resume the session."""
        print(
            f"\n{Colors.CYAN}Starting session for {Colors.BOLD}{self.agent_name}{Colors.RESET}{Colors.CYAN}...{Colors.RESET}"
        )

        # Use persistent config
        config = MemLearnConfig.default_persistent()

        # Create MemFS with persistence
        self.memfs = MemFS.for_agent(self.agent_name, config)
        self.tool_provider = OpenAIToolProvider(self.memfs, tool_prefix="memfs")

        # Initialize messages with system prompt
        system_prompt = get_system_prompt(self.agent_name)
        self.messages = [{"role": "system", "content": system_prompt}]

        print_success(f"Session started")
        print_info("Agent ID", self.memfs.config.agent_id or "N/A")
        print_info("Session ID", self.memfs.config.session_id or "N/A")
        print_info("Sandbox", self.memfs.sandbox.root_path)

        # Show any existing memory
        self._show_memory_status()

    def _show_memory_status(self):
        """Display current memory filesystem status."""
        print_subheader("Memory Status")

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
        """Execute a single chat turn."""
        self.turn_count += 1
        self.messages.append({"role": "user", "content": user_input})

        # Record in conversation history
        if self.memfs:
            self.memfs.append_conversation_message(
                {"role": "user", "content": user_input}
            )

        tools = self.tool_provider.get_tool_definitions()

        for round_num in range(max_tool_rounds):
            content, tool_calls, usage_info = self._stream_response(tools)

            # Build assistant message
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls and any(tc["id"] for tc in tool_calls):
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            # Record in conversation history
            if self.memfs and content:
                self.memfs.append_conversation_message(
                    {"role": "assistant", "content": content}
                )

            # Check if we have tool calls
            active_tool_calls = [tc for tc in tool_calls if tc.get("id")]
            if not active_tool_calls:
                self._display_usage(usage_info)
                break

            # Execute tool calls
            for tool_call in active_tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                # Execute and observe
                result_str = self.tool_provider.execute_tool(tool_name, arguments)
                try:
                    result_data = json.loads(result_str)
                except json.JSONDecodeError:
                    result_data = {"status": "unknown", "message": result_str[:100]}

                # Log to observer
                self.observer.log_tool_call(tool_name, arguments, result_data)

                # Add to messages
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result_str,
                    }
                )

    def _stream_response(self, tools: list) -> tuple[str, list, dict | None]:
        """Stream response from the model."""
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.messages,
            tools=tools,
            stream=True,
            stream_options={"include_usage": True},
        )

        full_content = ""
        tool_calls = []
        current_tool_call = None
        usage_info = None

        print(
            f"\n{Colors.BLUE}{Colors.BOLD}{self.agent_name}:{Colors.RESET} ",
            end="",
            flush=True,
        )

        for chunk in stream:
            if chunk.usage is not None:
                usage_info = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.index is not None:
                        while len(tool_calls) <= tc_delta.index:
                            tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )
                        current_tool_call = tool_calls[tc_delta.index]

                    if tc_delta.id:
                        current_tool_call["id"] = tc_delta.id

                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_call["function"][
                                "name"
                            ] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            current_tool_call["function"][
                                "arguments"
                            ] += tc_delta.function.arguments

        print()
        return full_content, tool_calls, usage_info

    def _display_usage(self, usage_info: dict | None):
        """Display token usage."""
        if not usage_info:
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
    session = PlaygroundSession(agent_name, observer)
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
    session2 = PlaygroundSession(agent_name, observer2)
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
    observer = MemLearnObserver(verbose=True)
    current_session = None

    while True:
        print_header("MemLearn Playground", Colors.CYAN)

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
            current_session = PlaygroundSession(arg, observer)
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
            current_session = PlaygroundSession(agent_name, observer)
            current_session.start()

            # Enter chat mode
            run_chat_mode(current_session, observer)
            current_session = None

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
    print(
        f"""
{Colors.DIM}{'─' * 60}
Chat mode with {session.agent_name}
Commands: /stats /inspect /memory /end /help
{'─' * 60}{Colors.RESET}
"""
    )

    try:
        while True:
            try:
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
                elif cmd == "help":
                    print(
                        f"""
{Colors.CYAN}Available commands:{Colors.RESET}
  /stats    - Show session statistics
  /inspect  - Inspect memory filesystem
  /memory   - Alias for /inspect
  /end      - End this session
  /help     - Show this help
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


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in a .env file or environment.")
        sys.exit(1)

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
            return

    # Run interactive menu
    interactive_menu()


if __name__ == "__main__":
    main()
