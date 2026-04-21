import os
import argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_chroma_path, get_embedding_function

DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--backend", choices=["openai", "ollama"], default=os.getenv("EMBEDDING_BACKEND", "openai"))
    parser.add_argument("--api-key", default="", help="API key to use for OpenAI embeddings.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database(get_chroma_path(args.backend))

    documents = load_documents()
    chunks = split_documents(documents)
    count = add_to_chroma(chunks, embedding_backend=args.backend, api_key=args.api_key)
    print(f"Done. Total chunks in DB: {count}")


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {DATA_PATH}/")
    return documents


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def add_to_chroma(chunks: list[Document], embedding_backend: str = "openai", api_key: str = ""):
    chroma_path = get_chroma_path(embedding_backend)
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function(backend=embedding_backend, api_key=api_key),
        collection_metadata={"hnsw:space": "cosine"},
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get(include=[])["ids"])
    print(f"Existing chunks in DB: {len(existing_ids)}")

    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks...")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
    else:
        print("No new chunks to add")

    return len(existing_ids) + len(new_chunks)


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["filename"] = os.path.basename(source)
        chunk.metadata["page_number"] = int(page) + 1
        last_page_id = current_page_id

    return chunks


def clear_database(chroma_path: str | None = None):
    import chromadb
    target_path = chroma_path or get_chroma_path()
    if os.path.exists(target_path):
        try:
            client = chromadb.PersistentClient(path=target_path)
            for collection in client.list_collections():
                client.delete_collection(collection.name)
        except Exception:
            pass
        shutil.rmtree(target_path, ignore_errors=True)
        print("Database cleared")


if __name__ == "__main__":
    main()
