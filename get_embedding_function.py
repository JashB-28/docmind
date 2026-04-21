import os
from dotenv import load_dotenv

load_dotenv()


def get_chroma_path(backend: str | None = None) -> str:
    selected_backend = (backend or os.getenv("EMBEDDING_BACKEND", "openai")).lower()
    return os.path.join("chroma", selected_backend)


def get_embedding_function(backend: str | None = None, api_key: str | None = None):
    selected_backend = (backend or os.getenv("EMBEDDING_BACKEND", "openai")).lower()

    if selected_backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY") if api_key is None else api_key,
    )
