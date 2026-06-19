import os
from dotenv import load_dotenv

load_dotenv()


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def get_embedding_function(backend: str | None = None, api_key: str | None = None):
    selected_backend = (backend or os.getenv("EMBEDDING_BACKEND", "openai")).lower()

    if selected_backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model, base_url=get_ollama_base_url())

    from langchain_openai import OpenAIEmbeddings
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or enter it "
            "in the sidebar before using OpenAI embeddings."
        )
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=key,
    )
