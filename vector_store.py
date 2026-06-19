"""Pinecone vector store helpers.

One serverless index per embedding backend (e.g. docmind-openai, docmind-ollama)
because each embedding model has its own vector dimension. A local JSON corpus
cache per backend keeps BM25 keyword search fast without scanning Pinecone.
"""

import json
import os
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from get_embedding_function import get_embedding_function

load_dotenv()

CORPUS_DIR = "bm25_corpus"

# Dimensions for common embedding models, so we only need a probe
# embedding call for models not listed here.
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


def get_pinecone_api_key() -> str:
    key = os.getenv("PINECONE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your .env file "
            "(create a free key at https://app.pinecone.io)."
        )
    return key


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=get_pinecone_api_key())


def get_backend(backend: str | None = None) -> str:
    return (backend or os.getenv("EMBEDDING_BACKEND", "openai")).lower()


def get_index_name(backend: str | None = None) -> str:
    prefix = os.getenv("PINECONE_INDEX_PREFIX", "docmind")
    return f"{prefix}-{get_backend(backend)}"


def _embedding_dimension(backend: str, embeddings) -> int:
    if backend == "ollama":
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    else:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    dimension = KNOWN_DIMENSIONS.get(model)
    if dimension is None:
        dimension = len(embeddings.embed_query("dimension probe"))
    return dimension


def _ensure_index(pc: Pinecone, index_name: str, backend: str, embeddings) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        return

    pc.create_index(
        name=index_name,
        dimension=_embedding_dimension(backend, embeddings),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def get_vector_store(backend: str | None = None, api_key: str = "") -> PineconeVectorStore:
    selected = get_backend(backend)
    embeddings = get_embedding_function(backend=selected, api_key=api_key or None)
    pc = get_pinecone_client()
    index_name = get_index_name(selected)
    _ensure_index(pc, index_name, selected, embeddings)
    return PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)


def get_existing_ids(backend: str | None = None) -> set[str]:
    """Return all vector ids currently stored in the backend's index."""
    pc = get_pinecone_client()
    index_name = get_index_name(backend)
    if index_name not in {idx["name"] for idx in pc.list_indexes()}:
        return set()

    index = pc.Index(index_name)
    ids: set[str] = set()
    # list() pages through ids on serverless indexes. If it fails (e.g. pod
    # index), fall back to re-upserting everything — upserts by id are
    # idempotent, so duplicates are impossible either way.
    try:
        for batch in index.list():
            ids.update(batch)
    except Exception:
        return set()
    return ids


def clear_database(backend: str | None = None) -> None:
    """Delete the backend's Pinecone index and its local BM25 corpus cache."""
    pc = get_pinecone_client()
    index_name = get_index_name(backend)
    if index_name in {idx["name"] for idx in pc.list_indexes()}:
        pc.delete_index(index_name)

    path = corpus_path(backend)
    if os.path.exists(path):
        os.remove(path)
    print("Database cleared")


# ── BM25 corpus cache ─────────────────────────────────────────────────────────

def corpus_path(backend: str | None = None) -> str:
    return os.path.join(CORPUS_DIR, f"{get_backend(backend)}.json")


def save_corpus(chunks: list[Document], backend: str | None = None) -> None:
    """Persist the full chunk corpus locally so BM25 search never hits Pinecone."""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    payload = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    with open(corpus_path(backend), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_corpus(backend: str | None = None) -> list[Document]:
    path = corpus_path(backend)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload
    ]
