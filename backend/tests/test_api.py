"""API wiring tests that need no external services or API keys."""

from api.main import app
from api.sessions import SessionManager
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_ok():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "active_sessions" in body


def test_documents_listing_empty_session():
    resp = client.get("/api/documents/does-not-exist")
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


def test_session_manager_isolates_and_evicts():
    mgr = SessionManager()
    a = mgr.get_or_create(None)
    b = mgr.get_or_create(None)
    assert a.session_id != b.session_id  # each session is isolated

    a.documents.append("doc.pdf")
    assert mgr.get(a.session_id).documents == ["doc.pdf"]

    # Force expiry and confirm eviction drops the session.
    a.last_seen = 0
    b.last_seen = 0
    assert mgr.evict_expired() == 2
    assert mgr.get(a.session_id) is None
