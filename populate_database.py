import os
import sys
import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store import clear_database, get_existing_ids, get_vector_store, save_corpus

DATA_PATH = "data"
UPSERT_BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--backend", choices=["openai", "ollama"], default=os.getenv("EMBEDDING_BACKEND", "openai"))
    parser.add_argument("--api-key", default="", help="API key to use for OpenAI embeddings.")
    args = parser.parse_args()
    try:
        if args.reset:
            print("Clearing Database")
            clear_database(args.backend)

        count = index_documents(backend=args.backend, api_key=args.api_key)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Done. Total chunks in DB: {count}")


def index_documents(backend: str = "openai", api_key: str = "") -> int:
    """Load PDFs from data/, chunk them, and upsert new chunks into Pinecone.

    Returns the total number of chunks in the index afterwards.
    """
    # Connect to Pinecone first so credential problems fail fast,
    # before any time is spent parsing PDFs.
    store = get_vector_store(backend=backend, api_key=api_key)

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No PDF pages found in {DATA_PATH}/. Add PDFs first.")

    chunks = split_documents(documents)
    chunks = calculate_chunk_ids(chunks)
    existing_ids = get_existing_ids(backend)
    print(f"Existing chunks in DB: {len(existing_ids)}")

    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks...")
        for start in range(0, len(new_chunks), UPSERT_BATCH_SIZE):
            batch = new_chunks[start:start + UPSERT_BATCH_SIZE]
            store.add_documents(batch, ids=[c.metadata["id"] for c in batch])
    else:
        print("No new chunks to add")

    # data/ always holds the full corpus, so rebuild the BM25 cache from scratch.
    save_corpus(chunks, backend=backend)

    return len(existing_ids | {c.metadata["id"] for c in new_chunks})


def load_documents():
    if not os.path.isdir(DATA_PATH):
        return []
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {DATA_PATH}/")
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


if __name__ == "__main__":
    main()
