"""Hybrid RAG query pipeline: Pinecone vector search + in-memory BM25.

Exposes both a blocking ``query_rag`` (used by compare/summarize/tests) and a
streaming ``stream_rag`` generator that yields the sources first and then the
answer token-by-token, which the API turns into Server-Sent Events.
"""

import argparse
import json
from typing import Iterator

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import settings
from .embeddings import get_ollama_base_url
from .vector_store import get_vector_store, load_corpus

PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about the user's documents using the context below.

Rules:
- Base your answer only on the context; do not use outside knowledge.
- Synthesize across excerpts: combine, list, or count information that is spread over multiple excerpts.
- If the context answers the question only partially or indirectly, give the best answer you can from
  what is there, and briefly note what the excerpts do not specify.
- Be specific: quote figures, names, and dates exactly as they appear.
- Only reply "I couldn't find that in the provided documents." if the context contains nothing
  related to the question at all.

Conversation so far:
{history}

Context from documents:
{context}

Question: {question}

Answer:"""

# Pinecone returns cosine similarity (higher = better). Scores at or above this
# value are treated as a 100% retrieval match for the confidence display.
FULL_CONFIDENCE_SIMILARITY = 0.75


def get_llm(model_name: str | None = None, api_key: str = ""):
    if model_name:
        if model_name.startswith("gpt"):
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model_name, api_key=api_key), "openai"
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=model_name, base_url=get_ollama_base_url()), "ollama"

    if settings.llm_backend.lower() == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=settings.ollama_llm_model,
            base_url=get_ollama_base_url(),
        ), "ollama"

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=api_key or settings.openai_api_key,
    ), "openai"


def similarity_to_confidence(score: float) -> int:
    return round(max(0.0, min(score / FULL_CONFIDENCE_SIMILARITY, 1.0)) * 100)


def chunk_key(doc) -> str:
    return doc.metadata.get("id") or doc.page_content[:50]


def _retrieve(
    query_text: str,
    namespace: str | None,
    embedding_backend: str,
    api_key: str,
    bm25_corpus: list[Document] | None,
    filename: str | None,
):
    """Run hybrid retrieval and return (unique_docs, vector_results)."""
    store = get_vector_store(backend=embedding_backend, api_key=api_key, namespace=namespace)

    # Vector search (semantic)
    search_kwargs = {"k": 10}
    if filename:
        search_kwargs["filter"] = {"filename": {"$eq": filename}}
    vector_results = store.similarity_search_with_score(query_text, **search_kwargs)
    vector_docs = [doc for doc, _ in vector_results]

    # BM25 search (keyword) over the in-memory corpus. Falls back to the on-disk
    # corpus for standalone CLI use when no corpus is passed in.
    corpus = bm25_corpus if bm25_corpus is not None else load_corpus(embedding_backend)
    if filename:
        corpus = [doc for doc in corpus if doc.metadata.get("filename") == filename]

    bm25_docs = []
    if corpus:
        bm25_retriever = BM25Retriever.from_documents(corpus)
        bm25_retriever.k = 8
        bm25_docs = bm25_retriever.invoke(query_text)

    # Merge and deduplicate — vector hits first, then BM25 additions
    seen_ids = set()
    unique_docs = []
    for doc in vector_docs + bm25_docs:
        cid = chunk_key(doc)
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_docs.append(doc)

    return unique_docs[:12], vector_results


def _build_sources(unique_docs, vector_results) -> list[dict]:
    score_map = {chunk_key(doc): score for doc, score in vector_results}
    sources = []
    seen = set()
    for doc in unique_docs:
        cid = chunk_key(doc)
        if cid in seen:
            continue
        seen.add(cid)

        # BM25-only hits have no vector score; show a neutral mid confidence.
        confidence = similarity_to_confidence(
            score_map.get(cid, FULL_CONFIDENCE_SIMILARITY * 0.5)
        )
        filename_meta = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        if isinstance(page, float):
            page = int(page)  # Pinecone stores all numbers as floats
        excerpt = doc.page_content

        sources.append({
            "filename": filename_meta,
            "page": page,
            "confidence": confidence,
            "chunk_id": cid,
            "excerpt": excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
        })
    return sources


def _build_prompt(unique_docs, query_text: str, chat_history: str) -> str:
    context_text = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        history=chat_history or "No prior conversation.",
        context=context_text,
        question=query_text,
    )


def query_rag(
    query_text: str,
    model_name: str | None = None,
    chat_history: str = "",
    api_key: str = "",
    embedding_backend: str = "openai",
    namespace: str | None = None,
    bm25_corpus: list[Document] | None = None,
    filename: str | None = None,
) -> dict:
    """Blocking hybrid RAG query. Returns {answer, sources, raw_sources}."""
    unique_docs, vector_results = _retrieve(
        query_text, namespace, embedding_backend, api_key, bm25_corpus, filename
    )

    if not unique_docs:
        return {
            "answer": "No relevant documents found. Index some PDFs first.",
            "sources": [],
            "raw_sources": [],
        }

    prompt = _build_prompt(unique_docs, query_text, chat_history)
    model, backend = get_llm(model_name=model_name, api_key=api_key)
    answer = model.invoke(prompt).content if backend == "openai" else model.invoke(prompt)

    return {
        "answer": answer,
        "sources": _build_sources(unique_docs, vector_results),
        "raw_sources": [chunk_key(doc) for doc in unique_docs],
    }


def stream_rag(
    query_text: str,
    model_name: str | None = None,
    chat_history: str = "",
    api_key: str = "",
    embedding_backend: str = "openai",
    namespace: str | None = None,
    bm25_corpus: list[Document] | None = None,
    filename: str | None = None,
) -> Iterator[tuple[str, object]]:
    """Yield ('sources', [...]) once, then ('token', str) repeatedly.

    The caller serializes these into Server-Sent Events.
    """
    unique_docs, vector_results = _retrieve(
        query_text, namespace, embedding_backend, api_key, bm25_corpus, filename
    )

    if not unique_docs:
        yield "sources", []
        yield "token", "No relevant documents found. Index some PDFs first."
        return

    yield "sources", _build_sources(unique_docs, vector_results)

    prompt = _build_prompt(unique_docs, query_text, chat_history)
    model, backend = get_llm(model_name=model_name, api_key=api_key)

    for chunk in model.stream(prompt):
        # ChatOpenAI yields message chunks; OllamaLLM yields plain strings.
        text = chunk.content if backend == "openai" else chunk
        if text:
            yield "token", text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--chat-history", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-backend", default="openai")
    parser.add_argument("--filename", default="", help="Restrict retrieval to one document.")
    args = parser.parse_args()

    result = query_rag(
        query_text=args.query_text,
        model_name=args.model_name or None,
        chat_history=args.chat_history,
        api_key=args.api_key,
        embedding_backend=args.embedding_backend,
        filename=args.filename or None,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
