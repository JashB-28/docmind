"""In-memory, ephemeral session store.

Each browser session owns a Pinecone namespace (its uploaded document vectors)
and an in-memory BM25 corpus. Nothing is written to a database or to disk, and
idle sessions are evicted after a TTL — so user documents do not outlive use.

This is single-process state. It is correct for a single Uvicorn worker (the
intended EC2 single-container deployment). Scaling to multiple workers/replicas
would require moving the corpus to a shared store (out of Phase 1 scope).
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from rag.config import settings
from rag.vector_store import clear_namespace


@dataclass
class Session:
    session_id: str
    backend: str = "openai"
    documents: list[str] = field(default_factory=list)  # filenames
    corpus: list[Document] = field(default_factory=list)  # BM25 corpus (in RAM)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session
            new_id = session_id or uuid.uuid4().hex
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    def clear(self, session_id: str) -> None:
        """Wipe a session's vectors (Pinecone namespace) and in-memory corpus."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            try:
                clear_namespace(backend=session.backend, namespace=session_id)
            except Exception:
                pass

    def evict_expired(self) -> int:
        """Drop sessions idle longer than the configured TTL. Returns count."""
        cutoff = time.time() - settings.session_ttl_minutes * 60
        expired: list[Session] = []
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if session.last_seen < cutoff:
                    expired.append(self._sessions.pop(sid))
        for session in expired:
            try:
                clear_namespace(backend=session.backend, namespace=session.session_id)
            except Exception:
                pass
        return len(expired)


sessions = SessionManager()
