"""Unit tests for Phase 3 retrieval logic — no external services or keys."""

from langchain_core.documents import Document
from rag.query import chunk_key, rrf_fuse
from rag.reranker import get_reranker


def _doc(cid: str, text: str = "x") -> Document:
    return Document(page_content=text, metadata={"id": cid})


def test_rrf_rewards_agreement_across_rankings():
    # "b" is ranked highly by both retrievers; "a" only by the first.
    vector = [_doc("a"), _doc("b"), _doc("c")]
    bm25 = [_doc("b"), _doc("d"), _doc("a")]

    fused = rrf_fuse([vector, bm25], k=60)
    ids = [chunk_key(d) for d in fused]

    assert ids[0] == "b"            # agreed-upon doc wins
    assert set(ids) == {"a", "b", "c", "d"}  # union, deduplicated


def test_rrf_handles_empty_rankings():
    only_vector = [_doc("a"), _doc("b")]
    fused = rrf_fuse([only_vector, []], k=60)
    assert [chunk_key(d) for d in fused] == ["a", "b"]
    assert rrf_fuse([[], []], k=60) == []


def test_reranker_disabled_by_default():
    # With RERANKER=none (default) no reranker is constructed, so the base
    # install never needs torch/cohere.
    get_reranker.cache_clear()
    assert get_reranker() is None
