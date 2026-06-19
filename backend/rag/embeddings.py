"""Embedding model factory for the OpenAI and Ollama backends."""

from .config import settings


def get_ollama_base_url() -> str:
    return settings.ollama_base_url


def get_embedding_function(backend: str | None = None, api_key: str | None = None):
    selected = (backend or settings.embedding_backend).lower()

    if selected == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    from langchain_openai import OpenAIEmbeddings

    key = api_key or settings.openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or pass it from "
            "the client before using OpenAI embeddings."
        )
    return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=key)
