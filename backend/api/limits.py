"""Simple, dependency-free abuse guards for the cost endpoints.

Two in-memory checks, applied to /query, /compare, and /documents/ingest:
  - per-IP sliding window  — stops one client from hammering the API;
  - site-wide daily cap     — bounds the worst-case Bedrock bill no matter what.

Both are process-local (correct for the single-container EC2 deploy). Use as a
FastAPI dependency: `dependencies=[Depends(rate_limit)]`.
"""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from rag.config import settings

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": "", "count": 0}


def _client_ip(request: Request) -> str:
    # Behind Caddy the socket peer is the proxy, so trust X-Forwarded-For's first
    # hop for the real client address.
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    now = time.time()
    ip = _client_ip(request)
    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_max_requests

    with _lock:
        # Per-IP sliding window.
        dq = _hits[ip]
        while dq and dq[0] < now - window:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(
                429, f"Too many requests — max {limit} per {window}s. Slow down a moment."
            )
        dq.append(now)

        # Site-wide daily cap (0 disables it).
        if settings.daily_request_cap > 0:
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if _day["date"] != today:
                _day["date"], _day["count"] = today, 0
            if _day["count"] >= settings.daily_request_cap:
                raise HTTPException(
                    429,
                    "This demo has hit its daily usage limit. Please try again tomorrow.",
                )
            _day["count"] += 1
