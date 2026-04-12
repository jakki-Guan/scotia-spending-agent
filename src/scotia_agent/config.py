"""Application config loaded from environment / .env file.

Single source of truth for all runtime configuration. Use `from
scotia_agent.config import settings` everywhere instead of reading
os.environ directly — that keeps config sources explicit, types
enforced, and tests able to override cleanly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — any OpenAI-compatible API (OpenRouter, DeepSeek direct,
    # Ollama local, Together, Azure, etc.). Provider is determined
    # entirely by base_url + model name.
    llm_api_key: str = Field(default="", description="OpenAI-compatible API key")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_model: str = Field(default="deepseek/deepseek-chat-v3:free")

    # Behavior flags
    llm_fallback_enabled: bool = Field(
        default=True,
        description="If False, categorize() returns 'uncategorized' instead of calling LLM",
    )


settings = Settings()
