"""Typed application configuration.

All runtime configuration is read once from the environment (or a local .env)
into a single ``Settings`` object, so the rest of the codebase never touches
``os.getenv`` directly. Import the shared ``settings`` instance from here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the repo root (…/backend/rag/config.py → repo root) so the .env is
# found no matter what directory the server is launched from. Real environment
# variables (e.g. those injected by docker-compose) still take precedence.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Only the real .env (templates _env / .env.example hold placeholders).
        env_file=_REPO_ROOT / ".env",
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

    # ── Amazon Bedrock ────────────────────────────────────────────────────────
    # Credentials come from the standard AWS chain (env vars or the EC2 instance
    # role) — no keys stored here. Enable model access in the Bedrock console and
    # make sure the IDs below are available in your region.
    aws_region: str = "us-east-1"
    # Current Claude models on Bedrock require a cross-region inference profile
    # (the "us."/"global." prefix) — raw model ids are not on-demand invokable.
    bedrock_llm_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"

    # ── S3 (optional, ephemeral source-document storage) ──────────────────────
    # Set s3_bucket to enable; blank = disabled (no-op). Uploaded PDFs are stored
    # under s3_prefix/<session_id>/ and deleted on clear / TTL (plus a bucket
    # lifecycle rule as a backstop). Presigned GET links expire after url_ttl.
    s3_bucket: str = ""
    s3_prefix: str = "uploads"
    s3_url_ttl_seconds: int = 3600

    # ── API / sessions ────────────────────────────────────────────────────────
    # Comma-separated list of allowed CORS origins, or "*" for any.
    cors_origins: str = "*"
    # Idle minutes before a session's vectors + corpus are evicted.
    session_ttl_minutes: int = 60
    # Hard limits on uploads to protect the box.
    max_upload_mb: int = 25
    max_pages_per_doc: int = 400

    # ── Public deployment controls ────────────────────────────────────────────
    # Which provider the UI selects by default, and whether to offer Ollama.
    # On the public site: DEFAULT_PROVIDER=bedrock, ENABLE_OLLAMA=false (no Ollama
    # server runs there). Locally, leave Ollama on.
    default_provider: str = "openai"
    enable_ollama: bool = True
    # Abuse guards on the cost endpoints (query/compare/ingest):
    #   per-IP sliding window + a site-wide daily cap. In-memory, no deps.
    rate_limit_max_requests: int = 20      # per IP ...
    rate_limit_window_seconds: int = 60    # ... per this many seconds
    daily_request_cap: int = 500           # site-wide ceiling per day (0 = off)

    # ── Observability (Phase 4) ───────────────────────────────────────────────
    log_level: str = "INFO"
    # Langfuse LLM tracing — optional. Leave keys blank to disable (no-op).
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Retrieval (Phase 3) ───────────────────────────────────────────────────
    # Candidates pulled from each retriever before fusion.
    vector_top_k: int = 10
    bm25_top_k: int = 10
    # RRF dampening constant (standard default is 60).
    rrf_k: int = 60
    # How many fused candidates to keep (the reranker, if any, sees these).
    fused_top_n: int = 12
    # Rewrite conversational follow-ups into standalone queries before retrieval.
    rewrite_queries: bool = True
    # Reranker backend: "none" | "cohere" | "local". The latter two need the
    # optional deps in backend/requirements-rerank.txt.
    reranker: str = "none"
    rerank_top_n: int = 5
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"
    local_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
