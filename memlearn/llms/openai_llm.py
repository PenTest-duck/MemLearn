"""
OpenAI LLM provider for MemLearn.

Provides chat completion capabilities using OpenAI's API for tasks
like summarization, reflection, and other generative operations.
"""

from __future__ import annotations

from openai import OpenAI

from memlearn.llms.base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """OpenAI-based LLM provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5-mini",
        default_max_tokens: int = 1024,
    ):
        """
        Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: The model to use for completions.
            default_max_tokens: Default maximum tokens for responses.
        """
        self.client = OpenAI(api_key=api_key)
        self._model = model
        self.default_max_tokens = default_max_tokens

    @property
    def model(self) -> str:
        """Return the model name being used."""
        return self._model

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 to 2.0).

        Returns:
            LLMResponse with the generated content.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.complete_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate a completion for a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 to 2.0).

        Returns:
            LLMResponse with the generated content.
        """
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature,
        )

        # Extract usage information
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=usage,
            raw_response=response,
        )

    def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        agent_name: str | None = None,
        session_context: str | None = None,
    ) -> str:
        """
        Summarize a conversation history.

        This method is optimized for summarizing agent conversation histories,
        capturing key decisions, outcomes, and learnings.

        Args:
            messages: The conversation messages to summarize.
            agent_name: Optional name of the agent for context.
            session_context: Optional additional context about the session.

        Returns:
            A structured summary of the conversation.
        """
        if not messages:
            return "No conversation to summarize."

        # Build conversation text
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Truncate very long messages for summarization
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            conversation_parts.append(f"{role}: {content}")

        conversation_text = "\n\n".join(conversation_parts)

        # Build system prompt for summarization
        agent_context = f" The conversation involves an agent named '{agent_name}'." if agent_name else ""
        extra_context = f" Additional context: {session_context}" if session_context else ""

        system_prompt = f"""You are an expert at summarizing agent conversations. Create a concise but comprehensive summary that captures:

1. **Primary Task/Goal**: What was the main objective of this conversation?
2. **Key Actions Taken**: What significant actions or decisions were made?
3. **Outcomes**: What was accomplished? Were there any errors or issues?
4. **Important Details**: Any specific files, data, or information that was worked with.
5. **Learnings**: Any insights or patterns that might be useful for future sessions.

Be concise but don't omit important details.{agent_context}{extra_context}"""

        prompt = f"Please summarize this conversation:\n\n{conversation_text}"

        response = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        return response.content
