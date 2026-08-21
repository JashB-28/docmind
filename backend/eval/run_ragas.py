"""RAGAS evaluation harness (two-phase, to dodge dependency conflicts).

Scores the RAG pipeline on a golden Q&A set using four reference metrics:

  - faithfulness        — is the answer grounded in the retrieved context?
                          (catches hallucination)
  - answer_relevancy    — does the answer actually address the question?
  - context_precision   — are the retrieved chunks relevant (low noise)?
  - context_recall      — did retrieval find everything needed for the answer?

RAGAS pins an older LangChain than the app, so the two cannot share one venv.
The harness is therefore split into two phases:

  generate — runs the app's RAG pipeline over the golden set and writes the
             answers + retrieved contexts to a rows file. Run in the APP venv.
  score    — loads a rows file and grades it with RAGAS. Run in an isolated
             eval venv that has ONLY ragas + datasets + langchain-openai.

Prereqs:
  1. Index the golden docs into the DEFAULT namespace:  python -m rag.ingest
  2. Fill backend/eval/golden.json with real questions + ground-truth answers.
  3. Set OPENAI_API_KEY (used for answering in generate, and for the judge in
     score).

A/B example (hybrid vs dense-only baseline):
  # app venv:
  python -m eval.run_ragas generate --retrieval hybrid --out eval/rows_hybrid.json
  python -m eval.run_ragas generate --retrieval dense  --out eval/rows_dense.json
  # eval venv:
  python -m eval.run_ragas score --rows eval/rows_hybrid.json
  python -m eval.run_ragas score --rows eval/rows_dense.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = EVAL_DIR / "golden.json"


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
        print(f"- answered: {question[:60]}")
    return rows


def cmd_generate(args: argparse.Namespace) -> None:
    """Phase 1 (app venv): produce answers + contexts for the golden set."""
    # Toggle retrieval mode for this run (dense = vector-only baseline).
    from rag.config import settings
    settings.use_bm25 = args.retrieval == "hybrid"
    print(f"Retrieval mode: {args.retrieval} (use_bm25={settings.use_bm25})")

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if any("REPLACE ME" in item.get("ground_truth", "") for item in golden):
        sys.exit("golden.json still has placeholder answers — fill it in first.")

    rows = build_rows(golden, args.backend)
    out = Path(args.out)
    out.write_text(json.dumps({"retrieval": args.retrieval, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {len(rows)} rows -> {out}")


def cmd_score(args: argparse.Namespace) -> None:
    """Phase 2 (eval venv): grade a rows file with RAGAS."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        sys.exit("OPENAI_API_KEY must be set — RAGAS uses an LLM to grade answers.")

    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows, tag = payload["rows"], payload.get("retrieval", "rows")

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

    dataset = EvaluationDataset.from_list(rows)
    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"Scoring '{tag}' ({len(rows)} rows) with RAGAS (judge LLM per item)…")
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)

    scores = {k: round(float(v), 4) for k, v in result._repr_dict.items()} \
        if hasattr(result, "_repr_dict") else dict(result)
    results_path = EVAL_DIR / f"results_{tag}.json"
    results_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print("\n=== RAGAS scores ===")
    for name, value in scores.items():
        print(f"  {name:22} {value}")
    print(f"\nSaved -> {results_path}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="App venv: answer the golden set, write a rows file.")
    gen.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    gen.add_argument("--backend", default="openai")
    gen.add_argument(
        "--retrieval",
        choices=["hybrid", "dense"],
        default="hybrid",
        help="hybrid = vector + BM25 + RRF (default); dense = vector-only baseline.",
    )
    gen.add_argument("--out", required=True, help="Where to write the rows JSON.")
    gen.set_defaults(func=cmd_generate)

    sc = sub.add_parser("score", help="Eval venv: grade a rows file with RAGAS.")
    sc.add_argument("--rows", required=True, help="Rows JSON produced by `generate`.")
    sc.add_argument("--min-faithfulness", type=float, default=0.0)
    sc.add_argument("--min-answer-relevancy", type=float, default=0.0)
    sc.add_argument("--min-context-precision", type=float, default=0.0)
    sc.add_argument("--min-context-recall", type=float, default=0.0)
    sc.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
