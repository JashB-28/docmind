# 🧠 DocMind — Fullstack RAG Document Assistant

A privacy-first, fullstack RAG application. Upload PDFs, ask questions, and get
**streamed** answers with **page-level citations** and **confidence scores**.
A **FastAPI** backend runs a hybrid (vector + BM25) retrieval pipeline over
**Pinecone**; a **React** frontend streams answers token-by-token over SSE.

Supports both OpenAI cloud models and local Ollama models.

---

## Architecture

```
                         ┌──────────────────────────┐
   React + Vite SPA  ─── │  FastAPI backend (/api)   │
   (streams tokens)      │                          │
        ▲                │  /documents/ingest       │
        │  SSE           │  /query   (SSE stream)   │
        │                │  /compare /documents …   │
        └──────────────  │                          │
                         └─────────┬────────────────┘
                                   │
          ┌──────────────┬──────────┴───┬──────────────┬──────────────┐
     Pinecone        In-RAM BM25     S3 (source     OpenAI / Bedrock
     (vectors,       + session       PDFs, presigned  / Ollama
     per-session     store           URLs, ephemeral) (LLM + embeddings)
     namespace)
```

Ingestion: `PDF → chunk → embed → upsert into the session's Pinecone namespace`
(the original PDF is also stored in S3 for presigned download, if enabled).
Query: `rewrite → vector search + BM25 → RRF fuse → (rerank) → stream answer + cite`.

### Retrieval pipeline

1. **History-aware rewrite** — conversational follow-ups ("and its revenue?") are
   condensed into a standalone query so retrieval isn't blind to prior turns.
2. **Hybrid retrieval** — Pinecone vector search (semantic) + in-memory BM25 (keyword).
3. **Reciprocal Rank Fusion (RRF)** — merges the two rankings by *rank position*,
   avoiding the trap of comparing cosine similarities against BM25 scores directly.
4. **Cross-encoder rerank** *(optional, confidence-adaptive)* — reorders fused
   candidates by true query-document relevance. Pluggable via `RERANKER`: `none`
   (default), `cohere` (hosted), or `local` (private cross-encoder on the box);
   enable with `backend/requirements-rerank.txt`. When on, it fires **adaptively**:
   queries whose top vector similarity already clears `RERANK_SIMILARITY_THRESHOLD`
   skip the reranker, so the cross-encoder's cost is spent only where retrieval
   is weak enough to need it.

### Privacy model

- Each browser gets a random **session id** that maps to its own **Pinecone
  namespace**. One session can never retrieve another session's documents.
- Sessions are held **in memory only** — no database, nothing written to local disk.
- Idle sessions (and their vectors) are **auto-deleted** after `SESSION_TTL_MINUTES`.
- Source PDFs in S3 are isolated per session and removed on clear/TTL, with a
  bucket **lifecycle rule** expiring any stragglers — ephemeral, not persistent.

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + TypeScript + Vite |
| Backend | FastAPI + Uvicorn, Server-Sent Events streaming |
| Orchestration | LangChain |
| Vector store | Pinecone (serverless), one namespace per session |
| Keyword search | BM25 (`rank_bm25`), in-memory per session |
| Fusion / rerank | Reciprocal Rank Fusion + optional cross-encoder (Cohere / local) |
| LLM | OpenAI (GPT-4o / 4o-mini), Amazon Bedrock (Claude Sonnet 4.5), or Ollama |
| Embeddings | OpenAI, Bedrock (Titan), or Ollama (`nomic-embed-text`) |
| Object storage | Amazon S3 — source PDFs + presigned downloads, lifecycle-expired |
| Packaging | Multi-stage Docker (one image serves API + SPA) |
| Observability | Structured JSON logs + request ids · optional Langfuse LLM tracing |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| CI | GitHub Actions — ruff lint, pytest, frontend build, Docker build |
| Tests | pytest (API wiring + LLM-as-judge RAG eval) |

---

## Project structure

```
backend/
├── rag/                # core RAG library
│   ├── config.py       # typed pydantic-settings config
│   ├── embeddings.py   # OpenAI / Bedrock / Ollama embeddings
│   ├── vector_store.py # Pinecone + per-session namespaces
│   ├── storage.py      # optional S3 store for source PDFs (presigned URLs)
│   ├── ingest.py       # load → chunk → upsert
│   ├── query.py        # rewrite → hybrid retrieve → RRF → rerank → (stream) answer
│   └── reranker.py     # pluggable cross-encoder rerank (none/cohere/local)
├── api/
│   ├── main.py         # FastAPI app, CORS, serves the built SPA
│   ├── schemas.py      # request/response models
│   ├── sessions.py     # in-memory, TTL-evicted session store
│   ├── observability.py# JSON logging, request-id middleware, Langfuse tracing
│   ├── limits.py       # per-IP rate limit + daily cap on cost endpoints
│   └── routers/        # health · documents · query (SSE) · compare
├── eval/               # RAGAS harness: golden.json + run_ragas.py
├── tests/              # test_api · test_retrieval · test_limits (no secrets) · test_rag
└── requirements.txt    # base · requirements-rerank.txt · requirements-eval.txt
frontend/               # Vite + React + TS chat UI (streams answers)
deploy/                 # docker-compose.prod.yml · docker-compose.behind-proxy.yml
                        # · Caddyfile · deploy.sh
docs/DEPLOY.md          # full deployment guide
.github/workflows/      # ci.yml (lint·test·build) · deploy.yml (Docker Hub → SSM)
Dockerfile              # multi-stage: build SPA → serve via FastAPI
docker-compose.yml      # local: build + run on :8000
```

---

## Quickstart (local dev)

### 1. Configure environment
```bash
cp .env.example .env      # then fill in OPENAI_API_KEY and PINECONE_API_KEY
```

### 2. Backend
```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r backend/requirements.txt
cd backend && uvicorn api.main:app --reload --port 8000
```

### 3. Frontend (separate terminal)
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api → :8000
```

Open http://localhost:5173, upload PDFs, and start asking.

---

## Docker (single container)

The image builds the React app and serves it from FastAPI — one process, one port.

```bash
docker compose up --build      # → http://localhost:8000
```

---

## Deploying on EC2 (beside other apps)

```bash
# on the instance
git clone <repo> && cd knowledgebaseai
cp .env.example .env && nano .env          # add your keys
docker compose up -d --build               # serves on :8000
```

Pick a host port that doesn't clash with your other services by editing the
`ports:` mapping in `docker-compose.yml` (e.g. `"8090:8000"`), then put it behind
your existing Nginx/reverse proxy. Tighten `CORS_ORIGINS` to your domain in `.env`.

**Production / CD.** `deploy/` holds compose files for both layouts — standalone
(`docker-compose.prod.yml`, own Caddy) and behind an existing reverse proxy
(`docker-compose.behind-proxy.yml`). `.github/workflows/deploy.yml` is a
push-button pipeline: GitHub Actions builds the image, pushes it to **Docker Hub**,
then triggers the EC2 box over **SSM** (keyless **OIDC**) to pull + restart.
**Caddy** terminates HTTPS via auto-renewing Let's Encrypt certs. Full guide in
[docs/DEPLOY.md](docs/DEPLOY.md).

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Liveness + config status |
| `POST` | `/api/documents/ingest` | Upload + index PDFs into the session namespace |
| `GET`  | `/api/documents/{session_id}` | List a session's indexed files |
| `DELETE` | `/api/documents/{session_id}` | Wipe a session's vectors |
| `POST` | `/api/query` | Ask a question — **streams** SSE: `sources`, `token`…, `done` |
| `POST` | `/api/compare` | Answer the same question across two documents |

Interactive docs at `/docs` when the server is running.

---

## Tests

```bash
cd backend
pytest tests/test_api.py tests/test_retrieval.py -v   # no secrets needed
pytest tests/test_rag.py -v                           # needs keys + indexed docs
```

---

## Observability, evaluation & CI

**Logging / tracing.** Every request emits a structured JSON log with a
`request_id`, path, status, and `duration_ms`; retrieval logs hit counts and
latency. On EC2 these go straight to journald/CloudWatch — no setup. Set the
`LANGFUSE_*` keys to additionally trace each LLM call (prompt, completion,
tokens, cost, latency) in the Langfuse UI; leave them blank and it's a no-op.

**RAGAS evaluation.** `backend/eval/run_ragas.py` scores the pipeline on a golden
Q&A set (`eval/golden.json`) across four metrics — faithfulness, answer
relevancy, context precision, context recall — and can fail under a threshold so
CI/you can gate on quality:
```bash
# in a separate venv (RAGAS has its own LangChain pins):
pip install -r backend/requirements.txt -r backend/requirements-eval.txt
cd backend && python -m rag.ingest            # index docs into the default namespace
python -m eval.run_ragas --min-faithfulness 0.7
```

**CI.** `.github/workflows/ci.yml` runs on every push/PR: ruff lint + pytest
(secret-free tests; RAG tests auto-skip), the frontend type-check/build, and a
Docker image build.

---

## How confidence scores work

Pinecone returns a cosine similarity per retrieved chunk. DocMind maps it to a
percentage, treating ≥ 0.75 similarity as a full match:

- 🟢 **70%+** — high confidence · 🟡 **40–69%** — medium · 🔴 **<40%** — low

---

## Skills demonstrated

`RAG` `FastAPI` `React` `TypeScript` `SSE Streaming` `LangChain` `Pinecone`
`Hybrid Search` `BM25` `RRF` `Cross-encoder Reranking` `Query Rewriting`
`Vector Embeddings` `OpenAI` `Amazon Bedrock` `Ollama` `Docker` `Multi-tenant namespaces`
`Observability` `Langfuse` `RAGAS Eval` `GitHub Actions CI/CD` `Structured Logging`
`AWS` `Amazon S3` `IAM Roles` `Presigned URLs` `Docker Hub` `OIDC` `SSM`
`Caddy / Let's Encrypt` `pytest` `Python`
