"""Client wrappers for calling OpenRouter models."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests import Response
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    """Raised when the language model API call fails."""


@dataclass
class LLMResult:
    sql: str
    raw: Dict[str, Any]


class OpenRouterLLM:
    """Simple wrapper around the OpenRouter chat completions endpoint."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 120) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is required to call the API."
            )
        self.timeout = timeout

    # Retry with exponential backoff for transient HTTP errors
    @retry(
        retry=retry_if_exception_type((requests.RequestException, LLMError)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    
    def generate(self, prompt: str, model: str) -> LLMResult:
        """Call OpenRouter to generate SQL for ``prompt`` using ``model``."""

        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        LOGGER.debug("Calling OpenRouter model %s", model)
        LOGGER.debug("Model's prompt %s", prompt)
        try:
            response: Response = requests.post(
                OPENROUTER_API_URL, json=payload, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            LOGGER.exception("OpenRouter request failed: %s", exc)
            raise

        if response.status_code != 200:
            LOGGER.error(
                "OpenRouter returned status %s: %s", response.status_code, response.text
            )
            raise LLMError(f"OpenRouter API error {response.status_code}")

        data = response.json()
        sql = self._extract_sql(data)

        LOGGER.debug("Received SQL: %s", sql)
        return LLMResult(sql=sql, raw=data)

    @staticmethod
    def _extract_sql(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("No choices returned from OpenRouter API.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise LLMError("No content in OpenRouter response.")

        return content.strip()


def safe_generate(prompt: str, model: str, api_key: Optional[str] = None) -> LLMResult:
    """Convenience function that wraps :class:`OpenRouterLLM` with logging."""

    client = OpenRouterLLM(api_key=api_key)
    try:
        return client.generate(prompt=prompt, model=model)
    except RetryError as exc:  # pragma: no cover - depends on API availability
        raise LLMError(f"OpenRouter request ultimately failed: {exc}") from exc
