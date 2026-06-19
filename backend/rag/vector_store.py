"""Pinecone vector store helpers.

One serverless index per embedding backend (e.g. docmind-openai, docmind-ollama)
because each embedding model has its own vector dimension. Within an index,
each user session gets its own **namespace**, so one session can never retrieve
another session's documents — the basis of the app's privacy model.

The BM25 keyword corpus is kept in memory by the API layer (see
``backend/api/sessions.py``); the optional on-disk helpers here exist only for
standalone CLI use against the default namespace.
"""

import json
import os
import time

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import settings
from .embeddings import get_embedding_function

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
    key = settings.pinecone_api_key.strip()
    if not key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your .env file "
            "(create a free key at https://app.pinecone.io)."
        )
    return key


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=get_pinecone_api_key())


def get_backend(backend: str | None = None) -> str:
    return (backend or settings.embedding_backend).lower()


def get_index_name(backend: str | None = None) -> str:
    return f"{settings.pinecone_index_prefix}-{get_backend(backend)}"


def _embedding_dimension(backend: str, embeddings) -> int:
    if backend == "ollama":
        model = settings.ollama_embedding_model
    else:
        model = settings.openai_embedding_model
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
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def get_vector_store(
    backend: str | None = None,
    api_key: str = "",
    namespace: str | None = None,
) -> PineconeVectorStore:
    selected = get_backend(backend)
    embeddings = get_embedding_function(backend=selected, api_key=api_key or None)
    pc = get_pinecone_client()
    index_name = get_index_name(selected)
    _ensure_index(pc, index_name, selected, embeddings)
    return PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embeddings,
        namespace=namespace or "",
    )


def get_existing_ids(backend: str | None = None, namespace: str | None = None) -> set[str]:
    """Return all vector ids stored in the backend's index for a namespace."""
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
        for batch in index.list(namespace=namespace or ""):
            ids.update(batch)
    except Exception:
        return set()
    return ids


def clear_namespace(backend: str | None = None, namespace: str | None = None) -> None:
    """Delete every vector belonging to a single session's namespace."""
    pc = get_pinecone_client()
    index_name = get_index_name(backend)
    if index_name not in {idx["name"] for idx in pc.list_indexes()}:
        return
    index = pc.Index(index_name)
    try:
        index.delete(delete_all=True, namespace=namespace or "")
    except Exception:
        # A namespace with no vectors raises; nothing to clear in that case.
        pass


def clear_database(backend: str | None = None) -> None:
    """Delete the backend's entire Pinecone index and its local corpus cache."""
    pc = get_pinecone_client()
    index_name = get_index_name(backend)
    if index_name in {idx["name"] for idx in pc.list_indexes()}:
        pc.delete_index(index_name)

    path = corpus_path(backend)
    if os.path.exists(path):
        os.remove(path)
    print("Database cleared")


# ── On-disk BM25 corpus cache (CLI / default namespace only) ──────────────────

def corpus_path(backend: str | None = None) -> str:
    return os.path.join(CORPUS_DIR, f"{get_backend(backend)}.json")


def save_corpus(chunks: list[Document], backend: str | None = None) -> None:
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
