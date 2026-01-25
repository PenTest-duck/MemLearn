import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-5.2"
REASONING_EFFORT = "medium"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a helpful, knowledgeable, and friendly AI assistant. You can:
- Answer questions on a wide variety of topics
- Help with analysis, writing, coding, and problem-solving
- Use available tools when appropriate to provide accurate information

Be concise but thorough. If you're unsure about something, say so. 
When using tools, explain what you're doing and interpret the results for the user."""

# Define available tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. 'London, UK' or 'New York, USA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather function - in production, this would call a real weather API."""
    # Simulated weather data
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": "12 km/h",
    }
    return weather_data


def calculate(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    import math

    # Define safe functions for evaluation
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": pow,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Only allow safe mathematical operations
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if tool_name == "get_weather":
        result = get_weather(**arguments)
    elif tool_name == "calculate":
        result = calculate(**arguments)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)


def stream_response(messages: list) -> tuple[str, list, dict | None]:
    """
    Stream a response from the model, handling tool calls.
    Returns: (assistant_message, tool_calls, usage_info)
    """
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        reasoning_effort=REASONING_EFFORT,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_content = ""
    tool_calls = []
    current_tool_call = None
    usage_info = None

    print("\n\033[94mAssistant:\033[0m ", end="", flush=True)

    for chunk in stream:
        # Check for usage info (comes in final chunk)
        if chunk.usage is not None:
            usage_info = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
            # Check for reasoning tokens in output_tokens_details
            if (
                hasattr(chunk.usage, "completion_tokens_details")
                and chunk.usage.completion_tokens_details
            ):
                details = chunk.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                    usage_info["reasoning_tokens"] = details.reasoning_tokens

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
                    # New tool call or continuing existing one
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
    """Display token usage information including reasoning tokens."""
    if not usage_info:
        return

    print("\n\033[90m--- Token Usage ---\033[0m")
    print(
        f"\033[90mPrompt: {usage_info.get('prompt_tokens', 'N/A')} | "
        f"Completion: {usage_info.get('completion_tokens', 'N/A')} | "
        f"Total: {usage_info.get('total_tokens', 'N/A')}\033[0m"
    )

    if "reasoning_tokens" in usage_info:
        print(f"\033[93mReasoning tokens: {usage_info['reasoning_tokens']}\033[0m")


def chat_turn(messages: list) -> list:
    """
    Execute a single chat turn, handling any tool calls.
    Returns the updated messages list.
    """
    content, tool_calls, usage_info = stream_response(messages)

    # Add assistant message to history
    assistant_message = {"role": "assistant", "content": content}
    if tool_calls and any(tc["id"] for tc in tool_calls):
        assistant_message["tool_calls"] = tool_calls
    messages.append(assistant_message)

    # Handle tool calls if any
    if tool_calls and any(tc["id"] for tc in tool_calls):
        for tool_call in tool_calls:
            if not tool_call["id"]:
                continue

            tool_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            print(f"\n\033[93m[Calling tool: {tool_name}({arguments})]\033[0m")

            result = execute_tool_call(tool_name, arguments)
            print(f"\033[93m[Tool result: {result}]\033[0m")

            # Add tool result to messages
            messages.append(
                {"role": "tool", "tool_call_id": tool_call["id"], "content": result}
            )

        # Get model's response to tool results
        content, _, usage_info = stream_response(messages)
        messages.append({"role": "assistant", "content": content})

    display_usage(usage_info)

    return messages


def main():
    print("\033[92m" + "=" * 50 + "\033[0m")
    print("\033[92m  OpenAI Agent - Interactive Chat\033[0m")
    print(f"\033[92m  Model: {MODEL_NAME} | Reasoning: {REASONING_EFFORT}\033[0m")
    print("\033[92m" + "=" * 50 + "\033[0m")
    print("\033[90mType your message and press Enter. Press Ctrl+C to exit.\033[0m")
    print("\033[90mAvailable tools: get_weather, calculate\033[0m\n")

    # Initialize conversation with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            # Get user input
            try:
                user_input = input("\033[92mYou:\033[0m ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Execute chat turn (handles streaming and tool calls)
            messages = chat_turn(messages)

            print()  # Extra spacing between turns

    except KeyboardInterrupt:
        print("\n\n\033[90mGoodbye!\033[0m")


if __name__ == "__main__":
    main()
