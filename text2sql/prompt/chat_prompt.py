"""Shared prompt structures for chat-based LLM calls."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatPrompt:
    """Container for separating system and user prompts."""

    system_prompt: str
    user_prompt: str

    def as_messages(self) -> list[dict[str, str]]:
        """Return the prompt as a list of role/content messages."""

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

