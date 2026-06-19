"""Hybrid RAG query pipeline with Phase 3 retrieval upgrades.

Pipeline per query:
  1. history-aware rewrite — condense conversational follow-ups into a
     standalone query so retrieval isn't blind to prior turns;
  2. hybrid retrieval — Pinecone vector search + in-memory BM25;
  3. Reciprocal Rank Fusion (RRF) — merge the two rankings by rank, not by
     incomparable raw scores;
  4. optional cross-encoder rerank — reorder fused candidates by true
     query-document relevance (see reranker.py).

Exposes a blocking ``query_rag`` and a streaming ``stream_rag`` generator.
"""

import argparse
import json
import logging
import time
from typing import Iterator

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import settings
from .embeddings import get_ollama_base_url
from .reranker import get_reranker
from .vector_store import get_vector_store, load_corpus

logger = logging.getLogger("rag.query")


def _config(callbacks):
    """LangChain run config carrying tracing callbacks, or None when unused."""
    return {"callbacks": callbacks} if callbacks else None

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

CONDENSE_TEMPLATE = """Given the conversation so far and a follow-up question, rewrite the \
follow-up into a standalone question that includes any context needed to retrieve relevant \
documents. Keep it concise and preserve the user's intent. If the question is already \
standalone, return it unchanged. Return only the rewritten question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""

# Pinecone returns cosine similarity (higher = better). Scores at or above this
# value are treated as a 100% retrieval match for the confidence display.
FULL_CONFIDENCE_SIMILARITY = 0.75


def _infer_provider(model_name: str | None) -> str:
    """Best-effort provider from a bare model name (CLI/legacy callers)."""
    if model_name and model_name.startswith("gpt"):
        return "openai"
    if model_name and (model_name.startswith(("anthropic.", "amazon.", "meta.", "us."))):
        return "bedrock"
    if model_name:
        return "ollama"
    return settings.llm_backend.lower()


def get_llm(model_name: str | None = None, api_key: str = "", provider: str | None = None):
    """Build an LLM for the given provider. Returns (model, provider)."""
    provider = (provider or _infer_provider(model_name)).lower()

    if provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model_name or settings.ollama_llm_model,
            base_url=get_ollama_base_url(),
        ), "ollama"

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # Credentials resolved via the standard AWS chain (env / instance role).
        return ChatBedrockConverse(
            model=model_name or settings.bedrock_llm_model,
            region_name=settings.aws_region,
        ), "bedrock"

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name or settings.openai_llm_model,
        api_key=api_key or settings.openai_api_key,
    ), "openai"


def _text_of(item) -> str:
    """Extract text from a chat message (OpenAI/Bedrock) or a raw string (Ollama)."""
    content = getattr(item, "content", item)
    # Bedrock's Converse API can return content as a list of typed blocks.
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


def _invoke_text(model, backend: str, prompt: str, callbacks=None) -> str:
    """Invoke a model and return plain text for any provider."""
    return _text_of(model.invoke(prompt, config=_config(callbacks)))


def similarity_to_confidence(score: float) -> int:
    return round(max(0.0, min(score / FULL_CONFIDENCE_SIMILARITY, 1.0)) * 100)


def chunk_key(doc) -> str:
    return doc.metadata.get("id") or doc.page_content[:50]


def condense_question(question: str, history: str, model, backend: str, callbacks=None) -> str:
    """Rewrite a conversational follow-up into a standalone retrieval query."""
    if not (settings.rewrite_queries and history.strip()):
        return question
    try:
        prompt = CONDENSE_TEMPLATE.format(history=history, question=question)
        rewritten = _invoke_text(model, backend, prompt, callbacks=callbacks).strip()
        return rewritten or question
    except Exception:
        # Never let rewriting failures break a query; fall back to the original.
        return question


def rrf_fuse(rankings: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal Rank Fusion: combine ranked lists by rank position.

    score(d) = sum over lists of 1 / (k + rank_in_list). This avoids comparing
    cosine similarities against BM25 scores, which live on different scales.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            cid = chunk_key(doc)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(cid, doc)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [docs[cid] for cid in ordered]


def _retrieve(
    retrieval_query: str,
    namespace: str | None,
    embedding_backend: str,
    api_key: str,
    bm25_corpus: list[Document] | None,
    filename: str | None,
):
    """Hybrid retrieval → RRF fusion → optional rerank.

    Returns (docs, vector_score_map, rerank_score_map).
    """
    started = time.perf_counter()
    store = get_vector_store(backend=embedding_backend, api_key=api_key, namespace=namespace)

    # Vector search (semantic)
    search_kwargs = {"k": settings.vector_top_k}
    if filename:
        search_kwargs["filter"] = {"filename": {"$eq": filename}}
    vector_results = store.similarity_search_with_score(retrieval_query, **search_kwargs)
    vector_docs = [doc for doc, _ in vector_results]
    vector_score_map = {chunk_key(doc): score for doc, score in vector_results}

    # BM25 search (keyword) over the in-memory corpus. Falls back to the on-disk
    # corpus for standalone CLI use when no corpus is passed in.
    corpus = bm25_corpus if bm25_corpus is not None else load_corpus(embedding_backend)
    if filename:
        corpus = [doc for doc in corpus if doc.metadata.get("filename") == filename]

    bm25_docs: list[Document] = []
    if corpus:
        bm25_retriever = BM25Retriever.from_documents(corpus)
        bm25_retriever.k = settings.bm25_top_k
        bm25_docs = bm25_retriever.invoke(retrieval_query)

    # Fuse the two rankings with RRF, then trim to the candidate pool.
    fused = rrf_fuse([vector_docs, bm25_docs], k=settings.rrf_k)[: settings.fused_top_n]

    # Optional cross-encoder rerank for final ordering.
    rerank_score_map: dict[str, float] = {}
    reranker = get_reranker()
    if reranker and fused:
        reranked = reranker(retrieval_query, fused)
        docs = [doc for doc, _ in reranked]
        rerank_score_map = {chunk_key(doc): score for doc, score in reranked}
    else:
        docs = fused

    logger.info(
        "retrieval",
        extra={"extra": {
            "vector_hits": len(vector_docs),
            "bm25_hits": len(bm25_docs),
            "fused": len(fused),
            "reranked": bool(rerank_score_map),
            "returned": len(docs),
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 1),
        }},
    )
    return docs, vector_score_map, rerank_score_map


def _build_sources(docs, vector_score_map, rerank_score_map) -> list[dict]:
    sources = []
    seen = set()
    for doc in docs:
        cid = chunk_key(doc)
        if cid in seen:
            continue
        seen.add(cid)

        # Prefer the reranker's relevance (0..1) when present; else fall back to
        # the vector similarity; BM25-only hits with neither get a neutral mid.
        if cid in rerank_score_map:
            confidence = round(rerank_score_map[cid] * 100)
        elif cid in vector_score_map:
            confidence = similarity_to_confidence(vector_score_map[cid])
        else:
            confidence = similarity_to_confidence(FULL_CONFIDENCE_SIMILARITY * 0.5)

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


def _build_prompt(docs, query_text: str, chat_history: str) -> str:
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
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
    callbacks=None,
) -> dict:
    """Blocking hybrid RAG query. Returns {answer, sources, contexts, raw_sources}."""
    model, backend = get_llm(model_name=model_name, api_key=api_key, provider=embedding_backend)
    retrieval_query = condense_question(query_text, chat_history, model, backend, callbacks)

    docs, vector_score_map, rerank_score_map = _retrieve(
        retrieval_query, namespace, embedding_backend, api_key, bm25_corpus, filename
    )

    if not docs:
        return {
            "answer": "No relevant documents found. Index some PDFs first.",
            "sources": [],
            "contexts": [],
            "raw_sources": [],
        }

    prompt = _build_prompt(docs, query_text, chat_history)
    answer = _invoke_text(model, backend, prompt, callbacks=callbacks)

    return {
        "answer": answer,
        "sources": _build_sources(docs, vector_score_map, rerank_score_map),
        # Full chunk texts (not truncated) — used by the RAGAS eval harness.
        "contexts": [doc.page_content for doc in docs],
        "raw_sources": [chunk_key(doc) for doc in docs],
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
    callbacks=None,
) -> Iterator[tuple[str, object]]:
    """Yield ('sources', [...]) once, then ('token', str) repeatedly.

    The caller serializes these into Server-Sent Events.
    """
    model, backend = get_llm(model_name=model_name, api_key=api_key, provider=embedding_backend)
    retrieval_query = condense_question(query_text, chat_history, model, backend, callbacks)

    docs, vector_score_map, rerank_score_map = _retrieve(
        retrieval_query, namespace, embedding_backend, api_key, bm25_corpus, filename
    )

    if not docs:
        yield "sources", []
        yield "token", "No relevant documents found. Index some PDFs first."
        return

    yield "sources", _build_sources(docs, vector_score_map, rerank_score_map)

    prompt = _build_prompt(docs, query_text, chat_history)
    for chunk in model.stream(prompt, config=_config(callbacks)):
        # Chat models (OpenAI/Bedrock) yield message chunks; Ollama yields strings.
        text = _text_of(chunk)
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
