"""RAGAS evaluation harness.

Scores the RAG pipeline on a golden Q&A set using four reference metrics:

  - faithfulness        — is the answer grounded in the retrieved context?
                          (catches hallucination)
  - answer_relevancy    — does the answer actually address the question?
  - context_precision   — are the retrieved chunks relevant (low noise)?
  - context_recall      — did retrieval find everything needed for the answer?

Prereqs:
  1. Index the documents your golden set asks about into the DEFAULT namespace:
        python -m rag.ingest                 # from backend/
  2. Fill in backend/eval/golden.json with real questions + ground-truth answers.
  3. Install eval deps in a separate venv (see requirements-eval.txt) and set
     OPENAI_API_KEY (RAGAS uses an LLM + embeddings to judge).

Run (from backend/):
    python -m eval.run_ragas
    python -m eval.run_ragas --min-faithfulness 0.7   # gate / fail under threshold

Exits non-zero if any metric falls below its threshold, so CI can gate on it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = EVAL_DIR / "golden.json"
RESULTS_PATH = EVAL_DIR / "results.json"


def build_rows(golden: list[dict], backend: str) -> list[dict]:
    """Run the pipeline over each golden item and shape RAGAS input rows."""
    from rag.query import query_rag

    rows = []
    for item in golden:
        question = item["question"]
        result = query_rag(question, embedding_backend=backend)
        rows.append({
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": result.get("contexts", []),
            "reference": item["ground_truth"],
        })
        print(f"· answered: {question[:60]}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--min-faithfulness", type=float, default=0.0)
    parser.add_argument("--min-answer-relevancy", type=float, default=0.0)
    parser.add_argument("--min-context-precision", type=float, default=0.0)
    parser.add_argument("--min-context-recall", type=float, default=0.0)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        sys.exit("OPENAI_API_KEY must be set — RAGAS uses an LLM to grade answers.")

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if any("REPLACE ME" in item.get("ground_truth", "") for item in golden):
        print("WARNING: golden.json still has placeholder answers — scores are meaningless.")

    # Imported lazily so the file lints/loads without the eval deps installed.
    from datasets import Dataset  # noqa: F401  (ensures the dep is present)
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = build_rows(golden, args.backend)
    dataset = EvaluationDataset.from_list(rows)

    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print("\nScoring with RAGAS (this calls the judge LLM per item)…")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)

    scores = {k: round(float(v), 4) for k, v in result._repr_dict.items()} \
        if hasattr(result, "_repr_dict") else dict(result)
    RESULTS_PATH.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print("\n=== RAGAS scores ===")
    for name, value in scores.items():
        print(f"  {name:22} {value}")
    print(f"\nSaved → {RESULTS_PATH}")

    thresholds = {
        "faithfulness": args.min_faithfulness,
        "answer_relevancy": args.min_answer_relevancy,
        "context_precision": args.min_context_precision,
        "context_recall": args.min_context_recall,
    }
    failures = [
        f"{name} {scores[name]} < {floor}"
        for name, floor in thresholds.items()
        if floor > 0 and name in scores and scores[name] < floor
    ]
    if failures:
        sys.exit("FAILED thresholds:\n  " + "\n  ".join(failures))
    print("\nAll thresholds met.")


if __name__ == "__main__":
    main()
