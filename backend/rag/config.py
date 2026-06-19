"""Typed application configuration.

All runtime configuration is read once from the environment (or a local .env)
into a single ``Settings`` object, so the rest of the codebase never touches
``os.getenv`` directly. Import the shared ``settings`` instance from here.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "_env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Backends ──────────────────────────────────────────────────────────────
    llm_backend: str = "openai"
    embedding_backend: str = "openai"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_prefix: str = "docmind"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "mistral"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── API / sessions ────────────────────────────────────────────────────────
    # Comma-separated list of allowed CORS origins, or "*" for any.
    cors_origins: str = "*"
    # Idle minutes before a session's vectors + corpus are evicted.
    session_ttl_minutes: int = 60
    # Hard limits on uploads to protect the box.
    max_upload_mb: int = 25
    max_pages_per_doc: int = 400

    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
