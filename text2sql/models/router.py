"""Client wrappers for calling OpenAI-compatible model providers."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion

LOGGER = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the language model API call fails."""


@dataclass
class LLMResult:
    sql: str
    raw: Dict[str, Any]


@dataclass
class RouterConfig:
    """Configuration required to initialise an OpenAI compatible router."""

    base_url: str
    api_key_env: str
    default_headers: Optional[Dict[str, str]] = None


# Import provider configs after defining RouterConfig to avoid circular imports.
from text2sql.models.provider.chatgpt import CHATGPT_CONFIG  # noqa: E402
from text2sql.models.provider.deepseek import DEEPSEEK_CONFIG  # noqa: E402
from text2sql.models.provider.openrouter import OPENROUTER_CONFIG  # noqa: E402

ROUTER_CONFIGS: Dict[str, RouterConfig] = {
    "openrouter": OPENROUTER_CONFIG,
    "deepseek": DEEPSEEK_CONFIG,
    "chatgpt": CHATGPT_CONFIG,
}


class OpenAIChatLLM:
    """Wrapper around OpenAI compatible chat completion endpoints."""

    def __init__(self, router: str, api_key: Optional[str] = None, timeout: int = 120) -> None:
        try:
            router_config = ROUTER_CONFIGS[router]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(
                f"Unsupported router '{router}'. Supported routers: {', '.join(sorted(ROUTER_CONFIGS))}."
            ) from exc

        resolved_api_key = api_key or os.getenv(router_config.api_key_env)
        if not resolved_api_key:
            raise EnvironmentError(
                f"{router_config.api_key_env} environment variable is required to call the {router} API."
            )

        LOGGER.debug("Initialising OpenAI client for router '%s'", router)

        client = OpenAI(
            api_key=resolved_api_key,
            base_url=router_config.base_url,
            default_headers=router_config.default_headers,
        )
        self.client = client.with_options(timeout=timeout)
        self.router = router

    def generate(self, prompt: str, model: str, max_tokens: Optional[int] = None) -> LLMResult:
        """Call the configured router to generate SQL for ``prompt`` using ``model``."""

        LOGGER.debug("Calling router '%s' with model %s", self.router, model)
        LOGGER.debug("Model prompt: %s", prompt)

        try:
            completion: ChatCompletion = self.client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        except OpenAIError as exc:  # pragma: no cover - network dependent
            LOGGER.exception("%s request failed: %s", self.router, exc)
            raise LLMError(f"{self.router} request failed") from exc

        sql = self._extract_sql(completion)
        LOGGER.debug("Received SQL: %s", sql)
        return LLMResult(sql=sql, raw=completion.model_dump())

    @staticmethod
    def _extract_sql(completion: ChatCompletion) -> str:
        if not completion.choices:
            raise LLMError("No choices returned from completion response.")

        message = completion.choices[0].message
        content = message.content
        if not content:
            raise LLMError("No content in completion response.")

        if isinstance(content, str):
            text_content = content
        else:
            # content may be a list of message parts; concatenate any text components
            text_parts = [
                getattr(part, "text", "")
                for part in content
                if getattr(part, "type", None) == "text" and getattr(part, "text", "")
            ]
            if not text_parts:
                raise LLMError("Completion content did not contain text parts.")
            text_content = "".join(text_parts)

        return text_content.strip()


def safe_generate(
    prompt: str,
    model: str,
    router: str = "openrouter",
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> LLMResult:
    """Convenience function that wraps :class:`OpenAIChatLLM` with logging."""

    client = OpenAIChatLLM(router=router, api_key=api_key)
    return client.generate(prompt=prompt, model=model, max_tokens=max_tokens)
