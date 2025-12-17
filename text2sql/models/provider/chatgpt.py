"""Router configuration for the OpenAI ChatGPT provider."""
from __future__ import annotations

from text2sql.models.router import RouterConfig

CHATGPT_CONFIG = RouterConfig(
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
)
