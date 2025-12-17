"""Router configuration for the OpenRouter provider (e.g., ChatGPT/OpenAI-compatible)."""
from __future__ import annotations

from text2sql.models.router import RouterConfig

OPENROUTER_CONFIG = RouterConfig(
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
)
