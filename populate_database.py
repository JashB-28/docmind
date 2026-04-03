import os
import argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    count = add_to_chroma(chunks)
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


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
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


def clear_database():
    import chromadb
    if os.path.exists(CHROMA_PATH):
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            for collection in client.list_collections():
                client.delete_collection(collection.name)
        except Exception:
            pass
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        print("Database cleared")


if __name__ == "__main__":
    main()
