"""DocMind API — FastAPI backend for the hybrid RAG pipeline.

Serves a JSON/SSE API under /api and, in production, the built React frontend
as static files from STATIC_DIR (default ./static).
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rag.config import settings

from api.observability import RequestContextMiddleware, configure_logging
from api.routers import documents, health, query
from api.sessions import sessions

STATIC_DIR = os.getenv("STATIC_DIR", "static")
EVICTION_INTERVAL_SECONDS = 300

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def evict_loop():
        while True:
            await asyncio.sleep(EVICTION_INTERVAL_SECONDS)
            try:
                sessions.evict_expired()
            except Exception:
                pass

    task = asyncio.create_task(evict_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="DocMind API", version="1.0.0", lifespan=lifespan)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")

# Serve the built frontend (SPA) when present. Registered last so /api wins.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
