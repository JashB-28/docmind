import os
from dotenv import load_dotenv

load_dotenv()


def get_embedding_function():
    backend = os.getenv("EMBEDDING_BACKEND", "openai").lower()

    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
