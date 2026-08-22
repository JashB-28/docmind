"""Deterministic keyword-retrieval analysis (zero API cost).

For each exact-term question, check whether the exact token appears in the
retrieved contexts (hit@k) and at what rank (-> MRR), comparing hybrid vs dense.
This measures keyword retrieval directly, without an LLM judge.
"""
import json
import re

# One distinctive exact token per question, aligned to golden_keyword.json order.
# Spaces/hyphens in a token are allowed to vary (PDF extraction splits them).
KEYS = [
    "CSMA 19", "DMV 141", "60V", "26,001", "1,000", "80-hour", "71%",
    "Kaggle", "Wasserstein", "jointly smooth", "Kakutani", "performative stability", "Bechavod",
    "Goodhart", "gaming", "separable", "Papadimitriou", "welfare",
    "NYUDepth", "depth estimation", "multi-task", "Mousavian", "Make3D",
    "Training-Free Retrieval", "R2R", "Matterport", "3 meters", "CLIP-Nav", "shortest path",
    "object goal navigation", "HM3D", "Path Length", "GLIP", "GPT-4o", "SAM", "Grounding dino",
]


def make_pattern(token: str) -> re.Pattern:
    # Allow optional spaces/hyphens between characters of the token so that
    # "GPT-4o" matches "GPT- 4o", "Matterport" matches "Matterport 3D", etc.
    esc = re.escape(token)
    esc = esc.replace(r"\ ", r"[\s\-]*").replace(r"\-", r"[\s\-]*")
    left = r"\b" if token[0].isalnum() else ""
    right = r"\b" if token[-1].isalnum() else ""
    return re.compile(left + esc + right, re.IGNORECASE)


def first_hit_rank(contexts, pat) -> int:
    for i, ctx in enumerate(contexts):
        if pat.search(ctx):
            return i + 1  # 1-indexed rank of first context containing the token
    return 0  # 0 = not found in any retrieved context


def analyze(path: str):
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    assert len(rows) == len(KEYS), f"{len(rows)} rows vs {len(KEYS)} keys"
    results = []
    for row, key in zip(rows, KEYS, strict=True):
        pat = make_pattern(key)
        rank = first_hit_rank(row["retrieved_contexts"], pat)
        results.append((key, rank))
    return results


h = analyze("backend/eval/rows_kw_hybrid.json")
d = analyze("backend/eval/rows_kw_dense.json")

print(f"{'token':24}{'hybrid':>8}{'dense':>8}   note")
hh = dd = 0
mrr_h = mrr_d = 0.0
for (k, rh), (_, rd) in zip(h, d, strict=True):
    hh += rh > 0
    dd += rd > 0
    mrr_h += (1.0 / rh) if rh else 0.0
    mrr_d += (1.0 / rd) if rd else 0.0
    note = ""
    if (rh > 0) != (rd > 0):
        note = "<-- HYBRID only" if rh else "<-- DENSE only"
    elif rh and rd and rh != rd:
        note = f"rank {rd}->{rh}"
    print(f"{k:24}{(rh or '-'):>8}{(rd or '-'):>8}   {note}")

n = len(KEYS)
print("\n=== keyword retrieval (exact-token hit in top-12 contexts) ===")
print(f"hybrid hit-rate: {hh}/{n} = {hh/n*100:.1f}%   MRR={mrr_h/n:.3f}")
print(f"dense  hit-rate: {dd}/{n} = {dd/n*100:.1f}%   MRR={mrr_d/n:.3f}")
if dd:
    print(f"hit-rate delta: {(hh-dd)/dd*100:+.1f}%   MRR delta: {(mrr_h-mrr_d)/mrr_d*100:+.1f}%")

# recall@k — for single-target questions, a "hit at rank <= k" == recall@k.
print("\n=== recall@k (exact-term chunk in top-k) ===")
hranks = [rh for _, rh in h]
dranks = [rd for _, rd in d]
for k in (5, 10, 12):
    rh = sum(1 for r in hranks if 0 < r <= k)
    rd = sum(1 for r in dranks if 0 < r <= k)
    delta = f"{(rh-rd)/rd*100:+.1f}%" if rd else "n/a"
    print(f"recall@{k:<2}  hybrid {rh}/{n}={rh/n*100:5.1f}%   dense {rd}/{n}={rd/n*100:5.1f}%   delta {delta}")
