"""
Mem0 Therapist Agent Demo

A therapist chatbot powered by:
- OpenAI Agents SDK for agent orchestration
- Mem0 MCP Server for persistent memory across sessions
- Exa for web search capabilities
- Full debug-level observability into agent operations

Usage:
    python playground/existing/mem0-demo.py

Environment variables required:
    - OPENAI_API_KEY: Your OpenAI API key
    - MEM0_API_KEY: Your Mem0 API key
    - EXA_API_KEY: Your Exa API key
    - MEM0_DEFAULT_USER_ID: (optional) User ID for Mem0, defaults to "therapist-user"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from exa_py import Exa

load_dotenv()

# Import OpenAI Agents SDK components
from agents import Agent, Runner, function_tool, RunHooks, RunContextWrapper, Tool
from agents.run_context import AgentHookContext
from agents.items import ModelResponse, TResponseInputItem
from agents.mcp import MCPServerStdio


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
# Debug Observability Hooks
# =============================================================================


class DebugHooks(RunHooks):
    """
    Custom RunHooks implementation for debug-level observability.
    Logs all agent lifecycle events with detailed information.
    """

    def __init__(self):
        self.turn_count = 0
        self.tool_calls = []
        self.llm_calls = []

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

    async def on_agent_start(
        self,
        context: AgentHookContext[Any],
        agent: Agent,
    ) -> None:
        self.turn_count += 1
        self._print_event(
            "AGENT START",
            Colors.CYAN,
            f"Agent: {Colors.WHITE}{agent.name}{Colors.RESET}\n"
            f"Turn: {Colors.WHITE}#{self.turn_count}{Colors.RESET}",
        )

    async def on_agent_end(
        self,
        context: AgentHookContext[Any],
        agent: Agent,
        output: Any,
    ) -> None:
        output_preview = (
            str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
        )
        self._print_event(
            "AGENT END",
            Colors.CYAN,
            f"Agent: {Colors.WHITE}{agent.name}{Colors.RESET}\n"
            f"Output preview: {Colors.DIM}{output_preview}{Colors.RESET}",
        )

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        # Count input items for display
        input_count = len(input_items) if input_items else 0
        self._print_event(
            "LLM START",
            Colors.BLUE,
            f"Agent: {Colors.WHITE}{agent.name}{Colors.RESET}\n"
            f"Model: {Colors.WHITE}{agent.model or 'default'}{Colors.RESET}\n"
            f"Input items: {Colors.WHITE}{input_count}{Colors.RESET}",
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        response: ModelResponse,
    ) -> None:
        # Try to extract token usage if available
        usage_info = ""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            usage_info = (
                f"Tokens - Input: {Colors.YELLOW}{getattr(usage, 'input_tokens', '?')}{Colors.RESET}, "
                f"Output: {Colors.YELLOW}{getattr(usage, 'output_tokens', '?')}{Colors.RESET}"
            )
        self._print_event(
            "LLM END",
            Colors.BLUE,
            (
                f"Agent: {Colors.WHITE}{agent.name}{Colors.RESET}\n" f"{usage_info}"
                if usage_info
                else f"Agent: {Colors.WHITE}{agent.name}{Colors.RESET}"
            ),
        )

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
    ) -> None:
        tool_name = getattr(tool, "name", str(tool))
        self.tool_calls.append(
            {
                "name": tool_name,
                "start_time": self._timestamp(),
            }
        )

        # Determine tool type and color
        if "memory" in tool_name.lower() or "mem0" in tool_name.lower():
            color = Colors.MAGENTA
            icon = "🧠"
        elif "search" in tool_name.lower() or "web" in tool_name.lower():
            color = Colors.GREEN
            icon = "🔍"
        else:
            color = Colors.YELLOW
            icon = "🔧"

        self._print_event(
            f"TOOL START {icon}",
            color,
            f"Tool: {Colors.WHITE}{tool_name}{Colors.RESET}",
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
        result: str,
    ) -> None:
        tool_name = getattr(tool, "name", str(tool))

        # Determine tool type and color
        if "memory" in tool_name.lower() or "mem0" in tool_name.lower():
            color = Colors.MAGENTA
            icon = "🧠"
        elif "search" in tool_name.lower() or "web" in tool_name.lower():
            color = Colors.GREEN
            icon = "🔍"
        else:
            color = Colors.YELLOW
            icon = "🔧"

        # Truncate result for display
        result_preview = result[:200] + "..." if len(result) > 200 else result
        # Clean up newlines for display
        result_preview = result_preview.replace("\n", " ")

        self._print_event(
            f"TOOL END {icon}",
            color,
            f"Tool: {Colors.WHITE}{tool_name}{Colors.RESET}\n"
            f"Result: {Colors.DIM}{result_preview}{Colors.RESET}",
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        self._print_event(
            "HANDOFF",
            Colors.YELLOW,
            f"From: {Colors.WHITE}{from_agent.name}{Colors.RESET}\n"
            f"To: {Colors.WHITE}{to_agent.name}{Colors.RESET}",
        )

    def print_summary(self):
        """Print a summary of the session's tool usage."""
        print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}Session Summary{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"  Total turns: {Colors.WHITE}{self.turn_count}{Colors.RESET}")
        print(f"  Tool calls: {Colors.WHITE}{len(self.tool_calls)}{Colors.RESET}")
        if self.tool_calls:
            print(f"\n  {Colors.DIM}Recent tool calls:{Colors.RESET}")
            for tc in self.tool_calls[-5:]:
                print(f"    - {tc['name']} at {tc['start_time']}")


# =============================================================================
# Exa Web Search Tool
# =============================================================================

# Initialize Exa client
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))


@function_tool
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
    try:
        result = exa_client.search(
            query,
            num_results=num_results,
            contents={"text": {"max_characters": 1000}},
        )

        if not result.results:
            return "No results found for the query."

        formatted_results = []
        for i, r in enumerate(result.results, 1):
            text_preview = (
                r.text[:500] + "..."
                if r.text and len(r.text) > 500
                else (r.text or "No content available")
            )
            formatted_results.append(
                f"{i}. **{r.title}**\n" f"   URL: {r.url}\n" f"   {text_preview}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"


# =============================================================================
# Therapist System Prompt
# =============================================================================

THERAPIST_SYSTEM_PROMPT = """You are Dr. Sage, a compassionate and experienced AI therapist. You provide a safe, non-judgmental space for people to explore their thoughts, feelings, and experiences.

## Your Approach

You practice an integrative therapeutic approach, drawing from:
- **Cognitive Behavioral Therapy (CBT)**: Helping identify and reframe unhelpful thought patterns
- **Person-Centered Therapy**: Providing unconditional positive regard and empathic understanding
- **Mindfulness-Based Approaches**: Encouraging present-moment awareness and acceptance
- **Solution-Focused Brief Therapy**: Focusing on strengths and solutions rather than problems

## Your Capabilities

You have access to special tools that enhance your ability to help:

1. **Memory Tools (via Mem0)**: You can save and recall important information about the person you're helping. Use these to:
   - Remember their name, background, and personal details they share
   - Track their progress, goals, and therapeutic journey
   - Recall previous conversations and insights
   - Build a continuous therapeutic relationship across sessions
   
2. **Web Search (via Exa)**: You can search for current information when helpful. Use this to:
   - Find relevant mental health resources
   - Look up specific therapeutic techniques
   - Find information about conditions or treatments when appropriate
   - Provide evidence-based recommendations

## Guidelines

1. **Always maintain therapeutic boundaries**: You are an AI assistant, not a replacement for human therapists. For serious mental health concerns, encourage seeking professional help.

2. **Use memory thoughtfully**: 
   - Save important personal details the user shares (with their implicit consent by sharing)
   - Search your memories at the start of conversations to recall context
   - Update memories when the user shares new information

3. **Be warm but professional**: Use a conversational, caring tone while maintaining appropriate boundaries.

4. **Practice active listening**: Reflect back what you hear, ask clarifying questions, and validate emotions.

5. **Empower the user**: Help them develop their own insights rather than prescribing solutions.

## Session Structure

- At the start of a new conversation, search your memories to recall who you're talking to
- If this is a new user, gently introduce yourself and ask how you can help
- Throughout the conversation, save important insights and information
- End sessions with a summary and encouragement when appropriate

Remember: Your goal is to provide a supportive, healing space while being transparent about your nature as an AI assistant."""


# =============================================================================
# Main Application
# =============================================================================


def print_header():
    """Print the application header."""
    print(f"\n{Colors.MAGENTA}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}  🧠 Mem0 Therapist Agent Demo{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'═' * 60}{Colors.RESET}")
    print(
        f"""
{Colors.WHITE}A therapist chatbot with persistent memory and web search.{Colors.RESET}

{Colors.DIM}Tools:{Colors.RESET}
  {Colors.MAGENTA}🧠 Mem0 MCP{Colors.RESET} - Persistent memory across sessions
  {Colors.GREEN}🔍 Exa{Colors.RESET}      - Web search for current information

{Colors.DIM}Commands:{Colors.RESET}
  Type your message to chat with the therapist
  Type {Colors.CYAN}/quit{Colors.RESET} or {Colors.CYAN}/exit{Colors.RESET} to end the session
  Type {Colors.CYAN}/debug{Colors.RESET} to toggle debug output
  Type {Colors.CYAN}/stats{Colors.RESET} to show session statistics
  Type {Colors.CYAN}/clear{Colors.RESET} to clear conversation history

{Colors.GRAY}{'─' * 60}{Colors.RESET}
"""
    )


async def run_therapist():
    """Main function to run the therapist agent."""
    print_header()

    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "MEM0_API_KEY", "EXA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(
            f"{Colors.RED}Error: Missing required environment variables:{Colors.RESET}"
        )
        for var in missing_vars:
            print(f"  - {var}")
        print(
            f"\n{Colors.DIM}Please set these in your .env file or environment.{Colors.RESET}"
        )
        return

    # Initialize debug hooks
    debug_hooks = DebugHooks()
    debug_enabled = True

    # Prepare Mem0 MCP server environment
    mem0_env = {
        "MEM0_API_KEY": os.getenv("MEM0_API_KEY"),
        "MEM0_DEFAULT_USER_ID": os.getenv("MEM0_DEFAULT_USER_ID", "therapist-user"),
    }

    print(f"{Colors.CYAN}Connecting to Mem0 MCP server...{Colors.RESET}")
    print(
        f"{Colors.DIM}(First run may take longer while packages are installed){Colors.RESET}"
    )

    # Create MCP server for Mem0
    # Note: Increase timeout to 60s to allow uvx to install packages on first run
    mem0_server = MCPServerStdio(
        name="mem0",
        params={
            "command": "uvx",
            "args": ["mem0-mcp-server"],
            "env": {**os.environ, **mem0_env},
        },
        cache_tools_list=True,
        client_session_timeout_seconds=60,  # Increased timeout for package installation
    )

    try:
        # Connect to the MCP server
        async with mem0_server:
            # List available tools for debugging
            tools = await mem0_server.list_tools()
            print(f"{Colors.GREEN}✓ Connected to Mem0 MCP server{Colors.RESET}")
            print(
                f"{Colors.DIM}  Available memory tools: {', '.join(t.name for t in tools)}{Colors.RESET}"
            )

            # Create the therapist agent
            therapist = Agent(
                name="dr-sage-therapist",
                instructions=THERAPIST_SYSTEM_PROMPT,
                tools=[web_search],  # Add Exa web search as a function tool
                mcp_servers=[mem0_server],  # Add Mem0 MCP server
                model="gpt-5.2",  # Use GPT-5.2 for best results
            )

            print(f"\n{Colors.GREEN}✓ Therapist agent ready!{Colors.RESET}")
            print(f"{Colors.GRAY}{'─' * 60}{Colors.RESET}\n")

            # Conversation loop
            conversation_history = []

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
                        debug_hooks.print_summary()
                        continue

                    elif cmd == "clear":
                        conversation_history = []
                        print(
                            f"{Colors.YELLOW}Conversation history cleared{Colors.RESET}\n"
                        )
                        continue

                    elif cmd == "help":
                        print(
                            f"""
{Colors.CYAN}Available commands:{Colors.RESET}
  /quit, /exit, /q  - End the session
  /debug            - Toggle debug output
  /stats            - Show session statistics
  /clear            - Clear conversation history
  /help             - Show this help message
"""
                        )
                        continue

                    else:
                        print(
                            f"{Colors.YELLOW}Unknown command: /{cmd}. Type /help for available commands.{Colors.RESET}\n"
                        )
                        continue

                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})

                # Run the agent
                print(
                    f"\n{Colors.BLUE}{Colors.BOLD}Dr. Sage:{Colors.RESET} ",
                    end="",
                    flush=True,
                )

                try:
                    # Run with streaming for better UX
                    result = Runner.run_streamed(
                        therapist,
                        conversation_history,
                        hooks=debug_hooks if debug_enabled else None,
                    )

                    full_response = ""
                    async for event in result.stream_events():
                        # Handle different event types
                        if event.type == "raw_response_event":
                            # Stream text output
                            if hasattr(event.data, "delta") and hasattr(
                                event.data.delta, "text"
                            ):
                                text = event.data.delta.text
                                if text:
                                    print(text, end="", flush=True)
                                    full_response += text

                    # Get final output (it's a property, not a method)
                    final_output = result.final_output
                    if final_output and not full_response:
                        print(final_output)
                        full_response = str(final_output)

                    print()  # New line after response

                    # Add assistant response to history
                    if full_response:
                        conversation_history.append(
                            {"role": "assistant", "content": full_response}
                        )

                except Exception as e:
                    print(f"\n{Colors.RED}Error: {str(e)}{Colors.RESET}")
                    if debug_enabled:
                        import traceback

                        print(f"{Colors.DIM}{traceback.format_exc()}{Colors.RESET}")

                print()  # Extra spacing between turns

    except Exception as e:
        print(f"{Colors.RED}Failed to start MCP server: {str(e)}{Colors.RESET}")
        import traceback

        print(f"{Colors.DIM}{traceback.format_exc()}{Colors.RESET}")

    # Print final summary
    if debug_hooks.tool_calls:
        debug_hooks.print_summary()


def main():
    """Entry point."""
    asyncio.run(run_therapist())


if __name__ == "__main__":
    main()
