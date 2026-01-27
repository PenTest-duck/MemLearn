"""
Letta Therapist Agent Demo

A therapist chatbot powered by:
- Letta SDK for stateful AI agents with persistent memory
- Exa for web search capabilities
- Built-in memory blocks for session continuity

Usage:
    python playground/existing/letta-demo.py

Environment variables required:
    - LETTA_API_KEY: Your Letta API key (or use self-hosted with LETTA_BASE_URL)
    - EXA_API_KEY: Your Exa API key
    - LETTA_BASE_URL: (optional) Base URL for self-hosted Letta server
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from exa_py import Exa
from letta_client import Letta

load_dotenv()


# =============================================================================
# ANSI Color Codes for Terminal Output
# =============================================================================


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

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
    BG_MAGENTA = "\033[45m"


# =============================================================================
# Debug Logger
# =============================================================================


class DebugLogger:
    """
    Logger for debug-level observability of agent operations.
    """

    def __init__(self):
        self.turn_count = 0
        self.tool_calls = []
        self.messages_sent = 0

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _print_event(self, event_type: str, color: str, details: str = ""):
        timestamp = self._timestamp()
        print(f"\n{Colors.GRAY}┌─ {timestamp} ─{'─' * 40}┐{Colors.RESET}")
        print(
            f"{Colors.GRAY}│{Colors.RESET} {color}{Colors.BOLD}[{event_type}]{Colors.RESET}"
        )
        if details:
            for line in details.split("\n"):
                print(f"{Colors.GRAY}│{Colors.RESET}   {line}")
        print(f"{Colors.GRAY}└{'─' * 55}┘{Colors.RESET}")

    def log_message_sent(self, message: str):
        """Log when a message is sent to the agent."""
        self.messages_sent += 1
        self.turn_count += 1
        preview = message[:80] + "..." if len(message) > 80 else message
        self._print_event(
            "MESSAGE SENT",
            Colors.CYAN,
            f"Turn: {Colors.WHITE}#{self.turn_count}{Colors.RESET}\n"
            f"Content: {Colors.DIM}{preview}{Colors.RESET}",
        )

    def log_tool_call(self, tool_name: str, tool_args: dict | None = None):
        """Log when a tool is called."""
        self.tool_calls.append(
            {
                "name": tool_name,
                "time": self._timestamp(),
            }
        )

        # Determine tool type and color
        if "memory" in tool_name.lower() or "archival" in tool_name.lower():
            color = Colors.MAGENTA
            icon = "🧠"
        elif "search" in tool_name.lower() or "web" in tool_name.lower():
            color = Colors.GREEN
            icon = "🔍"
        elif "conversation" in tool_name.lower():
            color = Colors.BLUE
            icon = "💬"
        else:
            color = Colors.YELLOW
            icon = "🔧"

        args_str = ""
        if tool_args:
            args_preview = str(tool_args)[:100]
            args_str = f"\nArgs: {Colors.DIM}{args_preview}{Colors.RESET}"

        self._print_event(
            f"TOOL CALL {icon}",
            color,
            f"Tool: {Colors.WHITE}{tool_name}{Colors.RESET}{args_str}",
        )

    def log_tool_result(self, tool_name: str, result: str):
        """Log when a tool returns a result."""
        # Determine tool type and color
        if "memory" in tool_name.lower() or "archival" in tool_name.lower():
            color = Colors.MAGENTA
            icon = "🧠"
        elif "search" in tool_name.lower() or "web" in tool_name.lower():
            color = Colors.GREEN
            icon = "🔍"
        elif "conversation" in tool_name.lower():
            color = Colors.BLUE
            icon = "💬"
        else:
            color = Colors.YELLOW
            icon = "🔧"

        result_preview = result[:200] + "..." if len(result) > 200 else result
        result_preview = result_preview.replace("\n", " ")

        self._print_event(
            f"TOOL RESULT {icon}",
            color,
            f"Tool: {Colors.WHITE}{tool_name}{Colors.RESET}\n"
            f"Result: {Colors.DIM}{result_preview}{Colors.RESET}",
        )

    def log_reasoning(self, reasoning: str):
        """Log agent's internal reasoning."""
        preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
        self._print_event(
            "REASONING 💭", Colors.GRAY, f"{Colors.DIM}{preview}{Colors.RESET}"
        )

    def log_response(self, content: str):
        """Log the final assistant response."""
        preview = content[:100] + "..." if len(content) > 100 else content
        self._print_event(
            "RESPONSE", Colors.BLUE, f"Preview: {Colors.DIM}{preview}{Colors.RESET}"
        )

    def print_summary(self):
        """Print a summary of the session."""
        print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}Session Summary{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"  Total turns: {Colors.WHITE}{self.turn_count}{Colors.RESET}")
        print(f"  Messages sent: {Colors.WHITE}{self.messages_sent}{Colors.RESET}")
        print(f"  Tool calls: {Colors.WHITE}{len(self.tool_calls)}{Colors.RESET}")
        if self.tool_calls:
            print(f"\n  {Colors.DIM}Recent tool calls:{Colors.RESET}")
            for tc in self.tool_calls[-5:]:
                print(f"    - {tc['name']} at {tc['time']}")


# =============================================================================
# Exa Web Search Tool Definition
# =============================================================================


def create_web_search_tool_definition():
    """Create the web search tool definition for Letta."""
    source_code = '''
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information using Exa.
    
    Use this tool when you need to:
    - Find current information about mental health topics
    - Look up therapeutic techniques or research
    - Find resources for the user
    - Get information about specific conditions or treatments
    
    Args:
        query: The search query to look up
        num_results: Number of results to return (default 5)
    
    Returns:
        A formatted string with search results including titles, URLs, and summaries
    """
    import os
    from exa_py import Exa
    
    try:
        exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
        result = exa_client.search(
            query,
            num_results=num_results,
            contents={"text": {"max_characters": 1000}},
        )

        if not result.results:
            return "No results found for the query."

        formatted_results = []
        for i, r in enumerate(result.results, 1):
            text_preview = r.text[:500] + "..." if r.text and len(r.text) > 500 else (r.text or "No content available")
            formatted_results.append(
                f"{i}. **{r.title}**\\n"
                f"   URL: {r.url}\\n"
                f"   {text_preview}\\n"
            )

        return "\\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"
'''
    return source_code


# =============================================================================
# Therapist Configuration
# =============================================================================

THERAPIST_PERSONA = """You are Dr. Sage, a compassionate and experienced AI therapist. You provide a safe, non-judgmental space for people to explore their thoughts, feelings, and experiences.

## Your Approach

You practice an integrative therapeutic approach, drawing from:
- **Cognitive Behavioral Therapy (CBT)**: Helping identify and reframe unhelpful thought patterns
- **Person-Centered Therapy**: Providing unconditional positive regard and empathic understanding
- **Mindfulness-Based Approaches**: Encouraging present-moment awareness and acceptance
- **Solution-Focused Brief Therapy**: Focusing on strengths and solutions rather than problems

## Your Capabilities

You have access to special tools:

1. **Memory Tools**: You can save and recall information in your memory blocks and archival memory. Use these to:
   - Remember the person's name, background, and personal details they share
   - Track their progress, goals, and therapeutic journey
   - Recall previous conversations and insights
   - Build a continuous therapeutic relationship across sessions
   
2. **Web Search**: You can search for current information when helpful. Use this to:
   - Find relevant mental health resources
   - Look up specific therapeutic techniques
   - Find information about conditions or treatments when appropriate

## Guidelines

1. **Always maintain therapeutic boundaries**: You are an AI assistant, not a replacement for human therapists. For serious mental health concerns, encourage seeking professional help.

2. **Use memory thoughtfully**: 
   - Save important personal details the user shares
   - Update your human memory block with key information about the person
   - Use archival memory for detailed notes and session summaries

3. **Be warm but professional**: Use a conversational, caring tone while maintaining appropriate boundaries.

4. **Practice active listening**: Reflect back what you hear, ask clarifying questions, and validate emotions.

5. **Empower the user**: Help them develop their own insights rather than prescribing solutions.

Remember: Your goal is to provide a supportive, healing space while being transparent about your nature as an AI assistant."""


HUMAN_MEMORY_TEMPLATE = """Information about the person I'm helping:

Name: Unknown
Background: Not yet shared
Current concerns: Not yet discussed
Goals: Not yet established
Key insights: None yet

I will update this as I learn more about them during our conversations."""


# =============================================================================
# Main Application
# =============================================================================


def print_header():
    """Print the application header."""
    print(f"\n{Colors.MAGENTA}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}  🧠 Letta Therapist Agent Demo{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'═' * 60}{Colors.RESET}")
    print(
        f"""
{Colors.WHITE}A therapist chatbot with stateful memory using Letta.{Colors.RESET}

{Colors.DIM}Features:{Colors.RESET}
  {Colors.MAGENTA}🧠 Core Memory{Colors.RESET}   - Persistent persona & human memory blocks
  {Colors.CYAN}📚 Archival{Colors.RESET}      - Long-term memory storage
  {Colors.GREEN}🔍 Web Search{Colors.RESET}   - Exa-powered information lookup

{Colors.DIM}Commands:{Colors.RESET}
  Type your message to chat with the therapist
  Type {Colors.CYAN}/quit{Colors.RESET} or {Colors.CYAN}/exit{Colors.RESET} to end the session
  Type {Colors.CYAN}/debug{Colors.RESET} to toggle debug output
  Type {Colors.CYAN}/stats{Colors.RESET} to show session statistics
  Type {Colors.CYAN}/memory{Colors.RESET} to view current memory blocks
  Type {Colors.CYAN}/new{Colors.RESET} to start fresh with a new agent

{Colors.GRAY}{'─' * 60}{Colors.RESET}
"""
    )


def get_or_create_agent(client: Letta, user_tag: str) -> tuple:
    """Get existing agent or create a new one for the user."""
    # Try to find existing agent for this user
    # Convert SyncArrayPage to list for indexing
    agents = list(client.agents.list(tags=[user_tag]))

    if agents:
        agent = agents[0]
        print(
            f"{Colors.GREEN}✓ Found existing agent: {agent.name} (ID: {agent.id[:8]}...){Colors.RESET}"
        )
        return agent, False

    # Create web search tool
    print(f"{Colors.CYAN}Creating web search tool...{Colors.RESET}")
    try:
        tool = client.tools.create(source_code=create_web_search_tool_definition())
        tool_names = [tool.name]
        print(f"{Colors.GREEN}✓ Created tool: {tool.name}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Could not create web search tool: {e}{Colors.RESET}")
        tool_names = []

    # Create new agent
    print(f"{Colors.CYAN}Creating new therapist agent...{Colors.RESET}")
    agent = client.agents.create(
        name="dr-sage-agent",
        model="OpenAI API Key/gpt-5.2",
        embedding="openai/text-embedding-3-small",
        memory_blocks=[
            {
                "label": "persona",
                "value": THERAPIST_PERSONA,
                "description": "The therapist's identity, approach, and guidelines. Read-only reference.",
            },
            {
                "label": "human",
                "value": HUMAN_MEMORY_TEMPLATE,
                "description": "Information about the person being helped. Update this as you learn about them.",
            },
        ],
        tools=tool_names,
        tags=[user_tag],
    )

    print(
        f"{Colors.GREEN}✓ Created new agent: {agent.name} (ID: {agent.id[:8]}...){Colors.RESET}"
    )
    return agent, True


def display_memory_blocks(client: Letta, agent_id: str):
    """Display the current memory blocks for the agent."""
    blocks = client.agents.blocks.list(agent_id)

    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}Current Memory Blocks{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")

    for block in blocks:
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}[{block.label}]{Colors.RESET}")
        if block.description:
            print(f"{Colors.DIM}Description: {block.description}{Colors.RESET}")
        print(f"{Colors.GRAY}{'─' * 40}{Colors.RESET}")
        # Truncate long values
        value = block.value if block.value else "(empty)"
        if len(value) > 500:
            value = value[:500] + "..."
        print(f"{Colors.WHITE}{value}{Colors.RESET}")

    print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}\n")


def process_response(response, debug_logger: DebugLogger, debug_enabled: bool) -> str:
    """Process the response from Letta and extract the assistant message."""
    assistant_content = ""

    for msg in response.messages:
        if msg.message_type == "assistant_message":
            assistant_content = msg.content
            if debug_enabled:
                debug_logger.log_response(assistant_content)

        elif msg.message_type == "reasoning_message":
            if debug_enabled and hasattr(msg, "reasoning") and msg.reasoning:
                debug_logger.log_reasoning(msg.reasoning)

        elif msg.message_type == "tool_call_message":
            if debug_enabled and hasattr(msg, "tool_call"):
                tool_call = msg.tool_call
                tool_name = getattr(tool_call, "name", "unknown")
                tool_args = getattr(tool_call, "arguments", None)
                debug_logger.log_tool_call(tool_name, tool_args)

        elif msg.message_type == "tool_return_message":
            if debug_enabled and hasattr(msg, "tool_return"):
                # Try to get the tool name from previous messages
                debug_logger.log_tool_result("tool", str(msg.tool_return)[:200])

    return assistant_content


def run_therapist():
    """Main function to run the therapist agent."""
    print_header()

    # Check required environment variables
    api_key = os.getenv("LETTA_API_KEY")
    base_url = os.getenv("LETTA_BASE_URL")
    exa_key = os.getenv("EXA_API_KEY")

    if not api_key and not base_url:
        print(
            f"{Colors.RED}Error: Missing LETTA_API_KEY or LETTA_BASE_URL{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}Set LETTA_API_KEY for Letta Cloud, or LETTA_BASE_URL for self-hosted.{Colors.RESET}"
        )
        return

    if not exa_key:
        print(
            f"{Colors.YELLOW}Warning: EXA_API_KEY not set. Web search will not work.{Colors.RESET}"
        )

    # Initialize Letta client
    print(f"{Colors.CYAN}Connecting to Letta...{Colors.RESET}")
    try:
        if base_url:
            client = Letta(base_url=base_url)
            print(
                f"{Colors.GREEN}✓ Connected to self-hosted Letta at {base_url}{Colors.RESET}"
            )
        else:
            client = Letta(api_key=api_key)
            print(f"{Colors.GREEN}✓ Connected to Letta Cloud{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Failed to connect to Letta: {e}{Colors.RESET}")
        return

    # Initialize debug logger
    debug_logger = DebugLogger()
    debug_enabled = True

    # User tag for agent persistence
    user_tag = os.getenv("LETTA_USER_TAG", "therapist-demo-user")

    # Get or create agent
    try:
        agent, is_new = get_or_create_agent(client, user_tag)
    except Exception as e:
        print(f"{Colors.RED}Failed to create/get agent: {e}{Colors.RESET}")
        import traceback

        print(f"{Colors.DIM}{traceback.format_exc()}{Colors.RESET}")
        return

    if is_new:
        print(
            f"{Colors.DIM}This is a new agent. It will remember you across sessions.{Colors.RESET}"
        )
    else:
        print(
            f"{Colors.DIM}Continuing previous session. The agent remembers past conversations.{Colors.RESET}"
        )

    print(f"\n{Colors.GREEN}✓ Therapist agent ready!{Colors.RESET}")
    print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}\n")

    # Conversation loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.GREEN}You:{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{Colors.CYAN}Session ended. Take care!{Colors.RESET}")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()

            if cmd in ("quit", "exit", "q"):
                print(
                    f"\n{Colors.CYAN}Session ended. Take care of yourself!{Colors.RESET}"
                )
                break

            elif cmd == "debug":
                debug_enabled = not debug_enabled
                status = "enabled" if debug_enabled else "disabled"
                print(f"{Colors.YELLOW}Debug output {status}{Colors.RESET}\n")
                continue

            elif cmd == "stats":
                debug_logger.print_summary()
                continue

            elif cmd == "memory":
                display_memory_blocks(client, agent.id)
                continue

            elif cmd == "new":
                # Delete old agent and create new one
                print(
                    f"{Colors.YELLOW}Deleting current agent and creating new one...{Colors.RESET}"
                )
                try:
                    client.agents.delete(agent.id)
                    agent, _ = get_or_create_agent(client, user_tag)
                    debug_logger = DebugLogger()  # Reset logger
                    print(
                        f"{Colors.GREEN}✓ Fresh start with new agent!{Colors.RESET}\n"
                    )
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.RESET}\n")
                continue

            elif cmd == "help":
                print(
                    f"""
{Colors.CYAN}Available commands:{Colors.RESET}
  /quit, /exit, /q  - End the session
  /debug            - Toggle debug output
  /stats            - Show session statistics
  /memory           - View current memory blocks
  /new              - Start fresh with a new agent
  /help             - Show this help message
"""
                )
                continue

            else:
                print(
                    f"{Colors.YELLOW}Unknown command: /{cmd}. Type /help for available commands.{Colors.RESET}\n"
                )
                continue

        # Log the message if debug is enabled
        if debug_enabled:
            debug_logger.log_message_sent(user_input)

        # Send message to agent
        print(
            f"\n{Colors.BLUE}{Colors.BOLD}Dr. Sage:{Colors.RESET} ", end="", flush=True
        )

        try:
            # Send message to Letta agent
            # Letta agents are stateful - they maintain their own conversation history
            response = client.agents.messages.create(
                agent_id=agent.id, messages=[{"role": "user", "content": user_input}]
            )

            # Process response
            assistant_content = process_response(response, debug_logger, debug_enabled)

            if assistant_content:
                print(assistant_content)
            else:
                print(f"{Colors.DIM}(No response content){Colors.RESET}")

        except Exception as e:
            print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}")
            if debug_enabled:
                import traceback

                print(f"{Colors.DIM}{traceback.format_exc()}{Colors.RESET}")

        print()  # Extra spacing between turns

    # Print final summary
    if debug_logger.tool_calls:
        debug_logger.print_summary()


def main():
    """Entry point."""
    run_therapist()


if __name__ == "__main__":
    main()
