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
              ┌────────────────────┼─────────────────────┐
        Pinecone (vectors,    In-RAM BM25 corpus     OpenAI / Ollama
        per-session namespace)  + session store        (LLM + embeddings)
```

Ingestion: `PDF → chunk → embed → upsert into the session's Pinecone namespace`.
Query: `rewrite → vector search + BM25 → RRF fuse → (rerank) → stream answer + cite`.

### Retrieval pipeline

1. **History-aware rewrite** — conversational follow-ups ("and its revenue?") are
   condensed into a standalone query so retrieval isn't blind to prior turns.
2. **Hybrid retrieval** — Pinecone vector search (semantic) + in-memory BM25 (keyword).
3. **Reciprocal Rank Fusion (RRF)** — merges the two rankings by *rank position*,
   avoiding the trap of comparing cosine similarities against BM25 scores directly.
4. **Cross-encoder rerank** *(optional)* — reorders fused candidates by true
   query-document relevance. Pluggable via `RERANKER`: `none` (default), `cohere`
   (hosted), or `local` (private cross-encoder on the box). Enable with
   `backend/requirements-rerank.txt`.

### Privacy model

- Each browser gets a random **session id** that maps to its own **Pinecone
  namespace**. One session can never retrieve another session's documents.
- Sessions are held **in memory only** — no database, nothing written to disk.
- Idle sessions (and their vectors) are **auto-deleted** after `SESSION_TTL_MINUTES`.

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
| LLM | OpenAI (GPT-4o / 4o-mini) or Ollama (Mistral, Llama3, …) |
| Embeddings | `text-embedding-3-small` or `nomic-embed-text` (Ollama) |
| Packaging | Multi-stage Docker (one image serves API + SPA) |
| Tests | pytest (API wiring + LLM-as-judge RAG eval) |

---

## Project structure

```
backend/
├── rag/                # core RAG library
│   ├── config.py       # typed pydantic-settings config
│   ├── embeddings.py   # OpenAI / Ollama embeddings
│   ├── vector_store.py # Pinecone + per-session namespaces
│   ├── ingest.py       # load → chunk → upsert
│   ├── query.py        # rewrite → hybrid retrieve → RRF → rerank → (stream) answer
│   └── reranker.py     # pluggable cross-encoder rerank (none/cohere/local)
├── api/
│   ├── main.py         # FastAPI app, CORS, serves the built SPA
│   ├── schemas.py      # request/response models
│   ├── sessions.py     # in-memory, TTL-evicted session store
│   └── routers/        # health · documents · query (SSE) · compare
├── tests/              # test_api.py (no secrets) · test_rag.py (LLM judge)
└── requirements.txt
frontend/               # Vite + React + TS chat UI (streams answers)
Dockerfile              # builds the SPA, serves it from FastAPI
docker-compose.yml
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
pytest tests/test_api.py -v          # no secrets needed
pytest tests/test_rag.py -v          # needs keys + indexed docs (LLM-as-judge)
```

---

## How confidence scores work

Pinecone returns a cosine similarity per retrieved chunk. DocMind maps it to a
percentage, treating ≥ 0.75 similarity as a full match:

- 🟢 **70%+** — high confidence · 🟡 **40–69%** — medium · 🔴 **<40%** — low

---

## Skills demonstrated

`RAG` `FastAPI` `React` `TypeScript` `SSE Streaming` `LangChain` `Pinecone`
`Hybrid Search` `BM25` `RRF` `Cross-encoder Reranking` `Query Rewriting`
`Vector Embeddings` `OpenAI` `Ollama` `Docker` `Multi-tenant namespaces`
`pytest` `Python`
