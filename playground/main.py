"""
MemLearn Playground - Research Agent Example

This example demonstrates a research agent that uses MemFS to:
1. Organize research findings into a structured memory system
2. Build knowledge over multiple interactions
3. Search and retrieve relevant information semantically
4. Learn and improve through reflection

The agent researches topics, takes notes, identifies patterns,
and builds a knowledge base that persists across interactions.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlearn import MemFS, MemLearnConfig, get_memfs_system_prompt
from memlearn.tools import OpenAIToolProvider

load_dotenv()

# Configuration
MODEL_NAME = "gpt-4o"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_system_prompt() -> str:
    """Build the system prompt using MemLearn's prompt provider."""
    # Get the MemFS system prompt component
    memfs_prompt = get_memfs_system_prompt(extended=True)

    return f"""You are a Research Agent - an AI assistant that excels at researching topics, organizing knowledge, and building a persistent memory system.

## Your Capabilities
- Research and analyze topics thoroughly
- Organize findings into your memory system
- Build knowledge that persists across conversations
- Search your memory for relevant past insights
- Learn from patterns and improve over time

{memfs_prompt}

## Research Process
When given a research task:
1. Search existing memory for relevant knowledge
2. Take working notes in /tmp/ as you research
3. Synthesize findings into permanent notes in /memory/
4. Connect new knowledge to existing insights
5. Summarize key learnings for the user

Be thorough, curious, and systematic. Your memory persists - build it wisely!"""


# Build the system prompt
SYSTEM_PROMPT = build_system_prompt()


def create_research_agent():
    """Create a research agent with MemFS integration."""

    # Configure MemFS for the research agent
    config = MemLearnConfig.from_env()
    config.agent_id = "research-agent"

    # Create MemFS instance
    memfs = MemFS(config=config)
    memfs.initialize()

    # Get tool provider
    tool_provider = OpenAIToolProvider(memfs, tool_prefix="memfs")

    return memfs, tool_provider


def stream_response(
    messages: list,
    tools: list,
) -> tuple[str, list, dict | None]:
    """
    Stream a response from the model, handling tool calls.
    Returns: (assistant_message, tool_calls, usage_info)
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_content = ""
    tool_calls = []
    current_tool_call = None
    usage_info = None

    print("\n\033[94mResearch Agent:\033[0m ", end="", flush=True)

    for chunk in stream:
        # Check for usage info (comes in final chunk)
        if chunk.usage is not None:
            usage_info = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Handle text content
        if delta.content:
            print(delta.content, end="", flush=True)
            full_content += delta.content

        # Handle tool calls
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                if tool_call_delta.index is not None:
                    while len(tool_calls) <= tool_call_delta.index:
                        tool_calls.append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    current_tool_call = tool_calls[tool_call_delta.index]

                if tool_call_delta.id:
                    current_tool_call["id"] = tool_call_delta.id

                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        current_tool_call["function"][
                            "name"
                        ] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        current_tool_call["function"][
                            "arguments"
                        ] += tool_call_delta.function.arguments

    print()  # Newline after response

    return full_content, tool_calls, usage_info


def display_usage(usage_info: dict | None):
    """Display token usage information."""
    if not usage_info:
        return

    print(
        f"\n\033[90m[Tokens: {usage_info.get('total_tokens', 'N/A')} "
        f"(prompt: {usage_info.get('prompt_tokens', 'N/A')}, "
        f"completion: {usage_info.get('completion_tokens', 'N/A')})]\033[0m"
    )


def chat_turn(
    messages: list,
    tools: list,
    tool_provider: OpenAIToolProvider,
    max_tool_rounds: int = 10,
) -> list:
    """
    Execute a chat turn with tool call handling.
    Returns the updated messages list.
    """
    for _ in range(max_tool_rounds):
        content, tool_calls, usage_info = stream_response(messages, tools)

        # Add assistant message
        assistant_message = {"role": "assistant", "content": content}
        if tool_calls and any(tc["id"] for tc in tool_calls):
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        # Check if we have tool calls to process
        active_tool_calls = [tc for tc in tool_calls if tc["id"]]
        if not active_tool_calls:
            display_usage(usage_info)
            break

        # Execute tool calls
        for tool_call in active_tool_calls:
            tool_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            # Display tool call
            print(f"\n\033[93m[{tool_name}]\033[0m")
            if arguments:
                # Show abbreviated arguments
                args_str = json.dumps(arguments, indent=2)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                print(f"\033[90m{args_str}\033[0m")

            # Execute tool
            result = tool_provider.execute_tool(tool_name, arguments)

            # Parse result for display
            try:
                result_data = json.loads(result)
                status = result_data.get("status", "unknown")
                message = result_data.get("message", "")

                if status == "success":
                    print(f"\033[92m✓ {message}\033[0m")
                else:
                    print(f"\033[91m✗ {message}\033[0m")
            except json.JSONDecodeError:
                print(
                    f"\033[90m{result[:200]}...\033[0m"
                    if len(result) > 200
                    else f"\033[90m{result}\033[0m"
                )

            # Add tool result to messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                }
            )

    return messages


def run_demo_tasks(memfs: MemFS, tool_provider: OpenAIToolProvider, tools: list):
    """Run a series of demo tasks to showcase MemFS capabilities."""

    demo_tasks = [
        "First, explore your memory filesystem. Use memfs_list to see what's available, then read the AGENTS.md file to understand how to use the system.",
        "Now let's do some research! Research and take notes on 'the attention mechanism in transformers'. Create an organized structure in /memory/research/ and write detailed notes. Include key concepts, how it works, and why it's important.",
        "Search your memory for anything related to 'attention' to verify your notes were saved correctly. Then add a new insight connecting attention mechanisms to human cognition.",
        "Create a summary document at /memory/summaries/transformers-attention.md that synthesizes your research. Reference your detailed notes.",
        "Finally, reflect on what you've learned in this session. Create a brief reflection note at /memory/reflections/ about the research process and any patterns you noticed.",
    ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\033[95m" + "=" * 60 + "\033[0m")
    print("\033[95m  MemLearn Research Agent Demo\033[0m")
    print("\033[95m" + "=" * 60 + "\033[0m")
    print("\033[90mThis demo shows the agent building knowledge using MemFS.\033[0m")
    print(
        "\033[90mWatch as it explores, researches, and organizes information.\033[0m\n"
    )

    for i, task in enumerate(demo_tasks, 1):
        print(f"\n\033[96m{'─' * 60}\033[0m")
        print(f"\033[96mTask {i}/{len(demo_tasks)}\033[0m")
        print(f"\033[96m{'─' * 60}\033[0m")
        print(f"\033[92mYou:\033[0m {task}\n")

        messages.append({"role": "user", "content": task})
        messages = chat_turn(messages, tools, tool_provider)

        # Brief pause between tasks
        input("\n\033[90m[Press Enter to continue to next task...]\033[0m")

    print("\n\033[95m" + "=" * 60 + "\033[0m")
    print("\033[95m  Demo Complete!\033[0m")
    print("\033[95m" + "=" * 60 + "\033[0m")
    print("\nThe agent has built a knowledge base in MemFS.")
    print("In a real application, this memory would persist across sessions.")


def interactive_mode(memfs: MemFS, tool_provider: OpenAIToolProvider, tools: list):
    """Run in interactive chat mode."""

    print("\033[92m" + "=" * 60 + "\033[0m")
    print("\033[92m  MemLearn Research Agent - Interactive Mode\033[0m")
    print("\033[92m" + "=" * 60 + "\033[0m")
    print("\033[90mChat with the research agent. It has access to MemFS for\033[0m")
    print("\033[90mstoring and retrieving knowledge. Type 'quit' to exit.\033[0m")
    print("\033[90mType 'demo' to run the automated demo.\033[0m\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "demo":
            run_demo_tasks(memfs, tool_provider, tools)
            # Reset messages after demo
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            continue

        messages.append({"role": "user", "content": user_input})
        messages = chat_turn(messages, tools, tool_provider)
        print()

    print("\n\033[90mGoodbye! Your MemFS knowledge has been preserved.\033[0m")


def main():
    """Main entry point."""
    print("\n\033[94mInitializing MemLearn Research Agent...\033[0m")

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\033[91mError: OPENAI_API_KEY environment variable not set.\033[0m")
        print("Please set your OpenAI API key in a .env file or environment.")
        sys.exit(1)

    # Create the research agent
    memfs, tool_provider = create_research_agent()
    tools = tool_provider.get_tool_definitions()

    print(f"\033[92m✓ MemFS initialized at: {memfs.sandbox.root_path}\033[0m")
    print(f"\033[92m✓ {len(tools)} tools available\033[0m\n")

    try:
        # Check command line args
        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
            run_demo_tasks(memfs, tool_provider, tools)
        else:
            interactive_mode(memfs, tool_provider, tools)
    finally:
        # Clean up
        print("\n\033[90mCleaning up MemFS...\033[0m")
        memfs.close()
        print("\033[92m✓ Done\033[0m")


if __name__ == "__main__":
    main()
