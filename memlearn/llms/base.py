"""
Abstract base class for MemLearn LLM providers.

LLM providers are used for tasks like summarization, reflection,
and other generative AI operations within MemFS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in the request."""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name being used."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def summarize(
        self,
        content: str,
        max_length: int | None = None,
        context: str | None = None,
    ) -> str:
        """
        Summarize the given content.

        This is a convenience method that uses complete() with a summarization prompt.
        Subclasses may override for provider-specific optimizations.

        Args:
            content: The content to summarize.
            max_length: Optional hint for maximum summary length.
            context: Optional context about what the content is.

        Returns:
            The summarized content as a string.
        """
        length_hint = ""
        if max_length:
            length_hint = f" Keep the summary under {max_length} words."

        context_hint = ""
        if context:
            context_hint = f" Context: {context}"

        system_prompt = (
            "You are a helpful assistant that creates concise, informative summaries. "
            "Focus on key points, decisions made, and important outcomes."
            f"{length_hint}"
        )

        prompt = f"Please summarize the following:{context_hint}\n\n{content}"

        response = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more focused summaries
        )

        return response.content
