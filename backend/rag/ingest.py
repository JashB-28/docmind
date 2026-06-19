"""PDF ingestion: load → chunk → embed → upsert into a Pinecone namespace.

``index_documents`` returns both the total vector count and the chunk list, so
callers (the API) can hold the chunks in memory for BM25 without ever writing
them to disk.
"""

import argparse
import os
import sys
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .vector_store import (
    clear_database,
    get_existing_ids,
    get_vector_store,
    save_corpus,
)

DATA_PATH = "data"
UPSERT_BATCH_SIZE = 64


@dataclass
class IngestResult:
    total_chunks: int
    chunks: list[Document]


def index_documents(
    data_path: str = DATA_PATH,
    backend: str = "openai",
    api_key: str = "",
    namespace: str | None = None,
) -> IngestResult:
    """Load PDFs from ``data_path``, chunk them, and upsert new chunks.

    Returns the total chunk count in the namespace plus the full chunk list.
    """
    # Connect to Pinecone first so credential problems fail fast,
    # before any time is spent parsing PDFs.
    store = get_vector_store(backend=backend, api_key=api_key, namespace=namespace)

    documents = load_documents(data_path)
    if not documents:
        raise RuntimeError(f"No PDF pages found in {data_path}/. Add PDFs first.")

    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)
    existing_ids = get_existing_ids(backend, namespace=namespace)
    print(f"Existing chunks in namespace: {len(existing_ids)}")

    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks...")
        for start in range(0, len(new_chunks), UPSERT_BATCH_SIZE):
            batch = new_chunks[start:start + UPSERT_BATCH_SIZE]
            store.add_documents(batch, ids=[c.metadata["id"] for c in batch])
    else:
        print("No new chunks to add")

    total = len(existing_ids | {c.metadata["id"] for c in new_chunks})
    return IngestResult(total_chunks=total, chunks=chunks)


def load_documents(data_path: str = DATA_PATH):
    if not os.path.isdir(data_path):
        return []
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {data_path}/")
    return documents


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata = sanitize_metadata(chunk.metadata)
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["filename"] = os.path.basename(source)
        try:
            chunk.metadata["page_number"] = int(page) + 1
        except (TypeError, ValueError):
            chunk.metadata["page_number"] = 1
        last_page_id = current_page_id

    return chunks


def sanitize_metadata(metadata: dict) -> dict:
    """Keep only Pinecone-compatible metadata values (no None, no nested objects)."""
    clean = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            clean[key] = value
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default=settings.embedding_backend,
    )
    parser.add_argument("--api-key", default="", help="API key for OpenAI embeddings.")
    args = parser.parse_args()
    try:
        if args.reset:
            print("Clearing Database")
            clear_database(args.backend)

        result = index_documents(backend=args.backend, api_key=args.api_key)
        # Standalone CLI use writes the corpus to disk so the CLI query path
        # (default namespace) has a BM25 corpus to load.
        save_corpus(result.chunks, backend=args.backend)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Done. Total chunks in DB: {result.total_chunks}")


if __name__ == "__main__":
    main()
