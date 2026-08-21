"""Retrieval-only A/B across configs (cheap: query embeddings only, no answer LLM,
no RAGAS judge). Measures recall@k and MRR of exact-term chunks, and how the
confidence-adaptive reranker gate (rerank only when top similarity < threshold)
trades reranking coverage for the MRR lift.
"""
import json
import re
import sys
import time

sys.path.insert(0, "backend")
from rag.config import settings  # noqa: E402
from rag.query import _retrieve  # noqa: E402
from rag.reranker import get_reranker  # noqa: E402

KEYS = [
    "CSMA 19", "DMV 141", "60V", "26,001", "1,000", "80-hour", "71%",
    "Kaggle", "Wasserstein", "jointly smooth", "Kakutani", "performative stability", "Bechavod",
    "Goodhart", "gaming", "separable", "Papadimitriou", "welfare",
    "NYUDepth", "depth estimation", "multi-task", "Mousavian", "Make3D",
    "Training-Free Retrieval", "R2R", "Matterport", "3 meters", "CLIP-Nav", "shortest path",
    "object goal navigation", "HM3D", "Path Length", "GLIP", "GPT-4o", "SAM", "Grounding dino",
]


def make_pattern(token: str) -> re.Pattern:
    esc = re.escape(token).replace(r"\ ", r"[\s\-]*").replace(r"\-", r"[\s\-]*")
    left = r"\b" if token[0].isalnum() else ""
    right = r"\b" if token[-1].isalnum() else ""
    return re.compile(left + esc + right, re.IGNORECASE)


PATS = [make_pattern(k) for k in KEYS]
GOLDEN = json.load(open("backend/eval/golden_keyword.json", encoding="utf-8"))
QUESTIONS = [g["question"] for g in GOLDEN]
assert len(QUESTIONS) == len(KEYS)

CONFIGS = [
    ("dense",            dict(use_bm25=False, rrf_k=60, reranker="none")),
    ("hybrid",           dict(use_bm25=True,  rrf_k=60, reranker="none")),
    ("rerank always-on", dict(use_bm25=True,  rrf_k=60, reranker="local", thr=1.0)),
    ("rerank gate@0.75", dict(use_bm25=True,  rrf_k=60, reranker="local", thr=0.75)),
    ("rerank gate@0.60", dict(use_bm25=True,  rrf_k=60, reranker="local", thr=0.60)),
    ("rerank gate@0.50", dict(use_bm25=True,  rrf_k=60, reranker="local", thr=0.50)),
]


def rank_for(contexts, pat) -> int:
    for i, c in enumerate(contexts):
        if pat.search(c):
            return i + 1
    return 0


def run_config(label, cfg):
    settings.use_bm25 = cfg["use_bm25"]
    settings.rrf_k = cfg["rrf_k"]
    settings.reranker = cfg["reranker"]
    if cfg["reranker"] != "none":
        settings.rerank_top_n = 12
        settings.rerank_similarity_threshold = cfg.get("thr", 1.0)
    get_reranker.cache_clear()

    ranks, fires = [], 0
    t0 = time.perf_counter()
    for q, pat in zip(QUESTIONS, PATS):
        docs, _, rr = _retrieve(q, None, "openai", "", None, None)
        if rr:  # rerank_score_map non-empty => the reranker actually fired
            fires += 1
        ranks.append(rank_for([d.page_content for d in docs], pat))
    dt = time.perf_counter() - t0

    n = len(ranks)
    r5 = sum(1 for r in ranks if 0 < r <= 5) / n
    r10 = sum(1 for r in ranks if 0 < r <= 10) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    return label, r5, r10, mrr, fires, dt


print(f"{'config':18}{'recall@5':>10}{'recall@10':>11}{'MRR':>8}{'fired':>8}{'sec':>7}")
rows = []
for label, cfg in CONFIGS:
    res = run_config(label, cfg)
    rows.append(res)
    label, r5, r10, mrr, fires, dt = res
    fired = f"{fires}/36" if cfg["reranker"] != "none" else "-"
    print(f"{label:18}{r5*100:>9.1f}%{r10*100:>10.1f}%{mrr:>8.3f}{fired:>8}{dt:>7.1f}")

base = next(r for r in rows if r[0] == "dense")
print("\nvs dense baseline (MRR):")
for label, r5, r10, mrr, fires, dt in rows:
    if label == "dense":
        continue
    print(f"  {label:18} MRR {(mrr-base[3])/base[3]*100:+6.1f}%   recall@10 {(r10-base[2])/base[2]*100:+5.1f}%")
