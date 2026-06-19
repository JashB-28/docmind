"""
RAG evaluation tests.

Uses OpenAI as the evaluator judge by default (set OPENAI_API_KEY in .env).
To switch to Ollama as judge, set EVAL_MODEL=ollama in .env.

These hit Pinecone + an LLM against the default namespace, so index some PDFs
first with:  python -m rag.ingest

To run:
    pytest backend/tests -v
"""

import os

import pytest
from dotenv import load_dotenv
from rag.query import query_rag

load_dotenv()

# These tests hit Pinecone and an LLM, so skip everything when keys are missing.
pytestmark = pytest.mark.skipif(
    not os.getenv("PINECONE_API_KEY", "").strip()
    or (
        os.getenv("LLM_BACKEND", "openai").lower() == "openai"
        and not os.getenv("OPENAI_API_KEY", "").strip()
    ),
    reason="PINECONE_API_KEY (and OPENAI_API_KEY for the openai backend) must be set in .env",
)

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
Does the actual response match the expected response in meaning?
Answer with only 'true' or 'false'.
"""


def get_evaluator():
    """Return the LLM used to judge test results."""
    eval_backend = os.getenv("EVAL_MODEL", "openai").lower()
    if eval_backend == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "mistral")), "ollama"
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    ), "openai"


def evaluate(question: str, expected_response: str) -> bool:
    result = query_rag(question)
    actual = result["answer"]

    eval_prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=actual,
    )

    evaluator, backend = get_evaluator()
    if backend == "openai":
        verdict = evaluator.invoke(eval_prompt).content.strip().lower()
    else:
        verdict = evaluator.invoke(eval_prompt).strip().lower()

    print(f"\nQ:        {question}")
    print(f"Expected: {expected_response}")
    print(f"Got:      {actual[:300]}")
    print(f"Scores:   {[s['confidence'] for s in result['sources']]}")

    if "true" in verdict:
        return True
    if "false" in verdict:
        return False
    raise ValueError(f"Evaluator returned unexpected verdict: {verdict!r}")


# ── Smoke tests — work with any document corpus ───────────────────────────────
class TestGenericRetrieval:

    def test_retrieval_returns_answer(self):
        result = query_rag("What is this document about?")
        assert result["answer"]
        assert len(result["answer"]) > 10

    def test_sources_are_returned(self):
        result = query_rag("Summarize the main topic.")
        assert len(result["sources"]) > 0

    def test_confidence_scores_in_range(self):
        result = query_rag("What are the key conclusions?")
        for src in result["sources"]:
            assert 0 <= src["confidence"] <= 100

    def test_unknown_query_handled_gracefully(self):
        result = query_rag("What is the population of Mars in 2099?")
        assert result["answer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
