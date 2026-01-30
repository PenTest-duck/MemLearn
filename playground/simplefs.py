#!/usr/bin/env python3
"""
SimpleFS Memory Agent

An OpenAI Agents SDK-based memory agent that uses a chroot-like jail
environment to store and retrieve memory via bash commands.

Each agent gets its own isolated filesystem at:
    ~/.memlearn/experimental/simplefs/<agent-name>/

The agent can execute arbitrary bash commands within this jail,
giving it full freedom to organize and manage its memory using
standard filesystem operations.
"""

import asyncio
import os
import subprocess
import shlex
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from agents import Agent, Runner, function_tool, SQLiteSession, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()

# Base directory for all agent filesystems
SIMPLEFS_BASE = Path.home() / ".memlearn" / "experimental" / "simplefs"
SIMPLEFS_BASE.mkdir(parents=True, exist_ok=True)

# Style for the prompt
PROMPT_STYLE = Style.from_dict({
    "prompt": "ansicyan bold",
    "agent_name": "ansigreen bold",
})


# ANSI color codes for observability
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def create_multiline_prompt_session(history_file: Path | None = None) -> PromptSession:
    """Create a prompt session that supports multi-line input.
    
    Enter: Insert newline
    Escape+Enter or Ctrl+D: Submit
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

    if history_file:
        return PromptSession(key_bindings=bindings, multiline=True, history=FileHistory(str(history_file)))
    return PromptSession(key_bindings=bindings, multiline=True)

# Thoughtful system prompt for memory-specialized agent
MEMORY_AGENT_SYSTEM_PROMPT = """You are a memory-specialized AI agent with persistent filesystem access.

## Your Core Purpose
You excel at remembering, organizing, and retrieving information across conversations. Your memory persists between sessions - what you write to your filesystem stays there forever until you delete it.

## Your Filesystem Environment
You have a dedicated filesystem directory that acts as your persistent brain. You can execute any bash command within this environment using the `bash` tool. Your working directory is always the root of your memory space.

## Memory Architecture Best Practices

### Organization Principles
1. **Hierarchical Structure**: Create meaningful folder hierarchies:
   - `/facts/` - Factual information about the user and world
   - `/preferences/` - User preferences and settings
   - `/conversations/` - Summaries of important conversations
   - `/tasks/` - Ongoing tasks and their states
   - `/knowledge/` - Domain knowledge and learnings
   - `/meta/` - Information about your own memory system

2. **File Naming**: Use descriptive, searchable names
   - Include dates for time-sensitive info: `2024-01-15_meeting_notes.md`
   - Use underscores for spaces: `user_preferences.md`
   - Use extensions: `.md` for text, `.json` for structured data

3. **File Content**: Write in a format you can easily parse later
   - Start files with a brief summary/purpose
   - Use markdown for readability
   - Include timestamps for when information was learned
   - Tag important information with keywords

### Memory Operations

**Storing New Information:**
- Before storing, check if related information exists
- Update existing files rather than creating duplicates
- Cross-reference related information

**Retrieving Information:**
- Use `find` and `grep` to locate relevant files
- Use `cat` to read file contents
- Use `ls -la` to explore directory structure

**Maintaining Memory:**
- Periodically consolidate related information
- Remove outdated or contradictory information
- Create index files in directories to summarize contents

## Bash Tool Usage

You can run any bash command. Common useful commands:
- `ls -la [path]` - List directory contents
- `cat [file]` - Read file contents
- `echo "content" > file` - Create/overwrite file
- `echo "content" >> file` - Append to file
- `mkdir -p [path]` - Create directories
- `find . -name "pattern"` - Find files
- `grep -r "pattern" .` - Search file contents
- `rm [file]` - Delete file
- `mv [src] [dst]` - Move/rename
- `tree` - Show directory structure (if available)

## Behavioral Guidelines

1. **Proactive Memory**: When you learn something important, store it without being asked
2. **Memory Retrieval**: When answering questions, check your memory first
3. **Honesty**: If you don't remember something, say so and check your files
4. **Organization**: Keep your memory tidy and well-organized
5. **Meta-awareness**: You can examine and reflect on your own memory structure

## First Session Setup

If this is a fresh memory space (empty directory), initialize a basic structure:
```bash
mkdir -p facts preferences conversations tasks knowledge meta
echo "# Memory System Initialized" > meta/README.md
echo "Created: $(date)" >> meta/README.md
```

Remember: Your filesystem IS your long-term memory. Use it wisely and consistently."""


def get_agent_dir(agent_name: str) -> Path:
    """Get the filesystem directory for an agent."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_name)
    return SIMPLEFS_BASE / safe_name


def get_session_db(agent_name: str) -> str:
    """Get the SQLite session database path for an agent."""
    agent_dir = get_agent_dir(agent_name)
    return str(agent_dir / ".session.db")


def list_agents() -> list[str]:
    """List all existing agents."""
    if not SIMPLEFS_BASE.exists():
        return []
    return [
        d.name for d in SIMPLEFS_BASE.iterdir() 
        if d.is_dir() and not d.name.startswith(".")
    ]


def create_agent_env(agent_name: str) -> Path:
    """Create the filesystem environment for a new agent."""
    agent_dir = get_agent_dir(agent_name)
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir


def create_bash_tool(agent_dir: Path):
    """Create a bash tool that executes commands within the agent's directory."""
    
    @function_tool
    def bash(command: str) -> str:
        """Execute a bash command within your memory filesystem.
        
        Your working directory is the root of your memory space.
        You can create files, directories, read, write, search - 
        anything you can do in bash.
        
        Args:
            command: The bash command to execute (e.g., 'ls -la', 'cat file.txt', 'echo "hello" > test.txt')
        
        Returns:
            The stdout and stderr output of the command, or an error message.
        """
        try:
            # Security: ensure we're always in the agent's directory
            # Using a restricted environment
            env = os.environ.copy()
            env["HOME"] = str(agent_dir)
            env["PWD"] = str(agent_dir)
            
            # Wrap command to ensure it runs from agent_dir
            # Use bash -c with cd to ensure proper directory
            wrapped_cmd = f'cd {shlex.quote(str(agent_dir))} && {command}'
            
            result = subprocess.run(
                ["bash", "-c", wrapped_cmd],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(agent_dir),
                env=env,
            )
            
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += f"[stderr]: {result.stderr}"
            
            if result.returncode != 0 and not output:
                output = f"Command exited with code {result.returncode}"
            
            return output if output else "(no output)"
            
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    return bash


def create_memory_agent(agent_name: str) -> tuple[Agent, Path]:
    """Create a memory agent with its filesystem environment."""
    agent_dir = create_agent_env(agent_name)
    bash_tool = create_bash_tool(agent_dir)
    
    agent = Agent(
        name=agent_name,
        instructions=MEMORY_AGENT_SYSTEM_PROMPT,
        tools=[bash_tool],
        model="gpt-5.2",
    )
    
    return agent, agent_dir


async def chat_with_agent(agent_name: str):
    """Interactive chat session with an agent."""
    agent, agent_dir = create_memory_agent(agent_name)
    session = SQLiteSession(agent_name, get_session_db(agent_name))
    
    # Setup multiline prompt with history
    history_file = agent_dir / ".prompt_history"
    prompt_session = create_multiline_prompt_session(history_file)
    
    # Track stats
    turn_count = 0
    
    print(f"\n{'='*60}")
    print(f"  Chatting with agent: {agent_name}")
    print(f"  Memory directory: {agent_dir}")
    print(f"{'='*60}")
    print("  Enter: newline | Esc+Enter or Ctrl+D: submit")
    print("  'exit'/'quit': menu | 'clear': reset history")
    print(f"{'='*60}\n")
    
    while True:
        try:
            user_input = await prompt_session.prompt_async(
                [("class:prompt", "You: ")],
                style=PROMPT_STYLE,
            )
            
            if not user_input.strip():
                continue
                
            if user_input.strip().lower() in ("exit", "quit"):
                print("\nReturning to menu...\n")
                break
                
            if user_input.strip().lower() == "clear":
                # Create a new session to clear history
                session = SQLiteSession(f"{agent_name}_{datetime.now().isoformat()}", get_session_db(agent_name))
                turn_count = 0
                print("\n[Conversation history cleared]\n")
                continue
            
            turn_count += 1
            
            # Run the agent with streaming
            result = Runner.run_streamed(
                agent,
                user_input,
                session=session,
            )
            
            # Track state for display
            printed_agent_prefix = False
            tool_call_count = 0
            
            # Stream events
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # Stream text tokens
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        if not printed_agent_prefix:
                            print(f"\n{Colors.BLUE}{Colors.BOLD}{agent_name}:{Colors.RESET} ", end="", flush=True)
                            printed_agent_prefix = True
                        print(event.data.delta, end="", flush=True)
                        
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        tool_call_count += 1
                        tool_name = event.item.raw_item.name if hasattr(event.item.raw_item, 'name') else "unknown"
                        print(f"\n{Colors.YELLOW}┌─ Tool Call #{tool_call_count}: {tool_name}{Colors.RESET}")
                        
                        # Try to show arguments
                        if hasattr(event.item.raw_item, 'arguments'):
                            args = event.item.raw_item.arguments
                            if args:
                                # Truncate long arguments
                                args_display = args if len(args) <= 200 else args[:200] + "..."
                                print(f"{Colors.YELLOW}│{Colors.RESET} {Colors.DIM}{args_display}{Colors.RESET}")
                        print(f"{Colors.YELLOW}└─{Colors.RESET}", end="", flush=True)
                        
                    elif event.item.type == "tool_call_output_item":
                        output = event.item.output
                        # Truncate long output
                        if len(output) > 300:
                            output = output[:300] + f"... ({len(event.item.output)} chars)"
                        print(f" {Colors.GREEN}→ {output}{Colors.RESET}")
                        printed_agent_prefix = False  # Reset so next text gets prefix
            
            # Ensure newline after streaming
            if printed_agent_prefix:
                print()
            
            # Show token usage
            usage = result.context_wrapper.usage
            print(f"{Colors.DIM}[Turn {turn_count} | Tokens: {usage.total_tokens} (in: {usage.input_tokens}, out: {usage.output_tokens}) | Tools: {tool_call_count}]{Colors.RESET}\n")
            
        except KeyboardInterrupt:
            print("\n\nReturning to menu...\n")
            break
        except EOFError:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}[Error: {e}]{Colors.RESET}\n")


def show_menu():
    """Display the main menu."""
    print("\n" + "="*60)
    print("  SimpleFS Memory Agent")
    print("="*60)
    print("\n  Available commands:")
    print("    1. List agents")
    print("    2. Create new agent")
    print("    3. Chat with agent")
    print("    4. Delete agent")
    print("    5. Show agent info")
    print("    0. Exit")
    print()


async def main():
    """Main entry point."""
    prompt_session = PromptSession()
    
    print("\n" + "="*60)
    print("  Welcome to SimpleFS Memory Agent")
    print("  Each agent has its own persistent filesystem for memory")
    print("="*60)
    
    while True:
        show_menu()
        
        try:
            choice = await prompt_session.prompt_async("Select option: ")
            choice = choice.strip()
            
            if choice == "0":
                print("\nGoodbye!\n")
                break
                
            elif choice == "1":
                # List agents
                agents = list_agents()
                if agents:
                    print("\nExisting agents:")
                    for i, name in enumerate(agents, 1):
                        agent_dir = get_agent_dir(name)
                        file_count = sum(1 for _ in agent_dir.rglob("*") if _.is_file() and not _.name.startswith("."))
                        print(f"  {i}. {name} ({file_count} files)")
                else:
                    print("\nNo agents found. Create one first!")
                    
            elif choice == "2":
                # Create new agent
                name = await prompt_session.prompt_async("Enter agent name: ")
                name = name.strip()
                if not name:
                    print("Agent name cannot be empty.")
                    continue
                if name in list_agents():
                    print(f"Agent '{name}' already exists.")
                    continue
                create_agent_env(name)
                print(f"\nAgent '{name}' created!")
                print(f"Memory directory: {get_agent_dir(name)}")
                
            elif choice == "3":
                # Chat with agent
                agents = list_agents()
                if not agents:
                    print("\nNo agents found. Create one first!")
                    continue
                    
                print("\nSelect an agent:")
                for i, name in enumerate(agents, 1):
                    print(f"  {i}. {name}")
                    
                selection = await prompt_session.prompt_async("Enter number or name: ")
                selection = selection.strip()
                
                # Try to parse as number
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(agents):
                        agent_name = agents[idx]
                    else:
                        print("Invalid selection.")
                        continue
                except ValueError:
                    # Treat as name
                    if selection in agents:
                        agent_name = selection
                    else:
                        print(f"Agent '{selection}' not found.")
                        continue
                
                await chat_with_agent(agent_name)
                
            elif choice == "4":
                # Delete agent
                agents = list_agents()
                if not agents:
                    print("\nNo agents found.")
                    continue
                    
                print("\nSelect agent to delete:")
                for i, name in enumerate(agents, 1):
                    print(f"  {i}. {name}")
                    
                selection = await prompt_session.prompt_async("Enter number or name: ")
                selection = selection.strip()
                
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(agents):
                        agent_name = agents[idx]
                    else:
                        print("Invalid selection.")
                        continue
                except ValueError:
                    if selection in agents:
                        agent_name = selection
                    else:
                        print(f"Agent '{selection}' not found.")
                        continue
                
                confirm = await prompt_session.prompt_async(
                    f"Are you sure you want to delete '{agent_name}'? This will erase all memory! (yes/no): "
                )
                if confirm.strip().lower() == "yes":
                    import shutil
                    agent_dir = get_agent_dir(agent_name)
                    shutil.rmtree(agent_dir)
                    print(f"\nAgent '{agent_name}' deleted.")
                else:
                    print("Cancelled.")
                    
            elif choice == "5":
                # Show agent info
                agents = list_agents()
                if not agents:
                    print("\nNo agents found.")
                    continue
                    
                print("\nSelect agent:")
                for i, name in enumerate(agents, 1):
                    print(f"  {i}. {name}")
                    
                selection = await prompt_session.prompt_async("Enter number or name: ")
                selection = selection.strip()
                
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(agents):
                        agent_name = agents[idx]
                    else:
                        print("Invalid selection.")
                        continue
                except ValueError:
                    if selection in agents:
                        agent_name = selection
                    else:
                        print(f"Agent '{selection}' not found.")
                        continue
                
                agent_dir = get_agent_dir(agent_name)
                
                print(f"\n{'='*60}")
                print(f"  Agent: {agent_name}")
                print(f"{'='*60}")
                print(f"  Directory: {agent_dir}")
                
                # Count files and directories
                files = list(agent_dir.rglob("*"))
                file_count = sum(1 for f in files if f.is_file() and not f.name.startswith("."))
                dir_count = sum(1 for f in files if f.is_dir())
                
                print(f"  Files: {file_count}")
                print(f"  Directories: {dir_count}")
                
                # Show directory structure (top level)
                print(f"\n  Top-level contents:")
                for item in sorted(agent_dir.iterdir()):
                    if item.name.startswith("."):
                        continue
                    if item.is_dir():
                        subcount = sum(1 for _ in item.rglob("*") if _.is_file())
                        print(f"    📁 {item.name}/ ({subcount} files)")
                    else:
                        size = item.stat().st_size
                        print(f"    📄 {item.name} ({size} bytes)")
                        
            else:
                print("Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except EOFError:
            print("\n\nGoodbye!\n")
            break


if __name__ == "__main__":
    asyncio.run(main())
