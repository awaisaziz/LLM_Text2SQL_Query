"""Router configuration for the DeepSeek provider."""
from __future__ import annotations

from text2sql.models.router import RouterConfig

DEEPSEEK_CONFIG = RouterConfig(
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
)
