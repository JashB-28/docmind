"""Pluggable cross-encoder reranking.

Reranking reorders the fused candidate chunks by true query–document relevance,
which is the single biggest retrieval-quality lever. It is **opt-in**: the base
install ships no reranker (``RERANKER=none``) so the image stays lean. Enable a
backend by installing backend/requirements-rerank.txt and setting RERANKER.

A reranker is a callable: ``(query, docs) -> list[(Document, score)]`` sorted by
descending relevance, where score is normalized to 0..1.
"""

import math
from functools import lru_cache
from typing import Callable

from langchain_core.documents import Document

from .config import settings

Reranker = Callable[[str, list[Document]], list[tuple[Document, float]]]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """Return the configured reranker, or None when reranking is disabled.

    Cached so heavyweight local models load only once per process.
    """
    backend = settings.reranker.lower()
    if backend == "cohere":
        return _build_cohere_reranker()
    if backend == "local":
        return _build_local_reranker()
    return None


def _build_cohere_reranker() -> Reranker:
    import cohere

    if not settings.cohere_api_key.strip():
        raise RuntimeError("RERANKER=cohere but COHERE_API_KEY is not set.")
    client = cohere.Client(settings.cohere_api_key)
    model = settings.cohere_rerank_model

    def rerank(query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        if not docs:
            return []
        response = client.rerank(
            model=model,
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(settings.rerank_top_n, len(docs)),
        )
        # Cohere relevance_score is already 0..1.
        return [(docs[r.index], float(r.relevance_score)) for r in response.results]

    return rerank


def _build_local_reranker() -> Reranker:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.local_reranker_model)

    def rerank(query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        if not docs:
            return []
        scores = model.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        # Cross-encoder logits are unbounded; squash to 0..1 for confidence.
        top = ranked[: settings.rerank_top_n]
        return [(doc, _sigmoid(float(score))) for doc, score in top]

    return rerank
