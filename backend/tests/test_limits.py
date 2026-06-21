"""Rate-limit guard tests — no external services."""

import api.limits as limits
from rag.config import settings


class _Req:
    """Minimal stand-in for a Starlette Request."""

    def __init__(self, ip: str):
        self.headers = {"x-forwarded-for": ip}
        self.client = type("C", (), {"host": ip})()


def setup_function():
    limits._hits.clear()
    limits._day.update(date="", count=0)


def test_per_ip_window_blocks_after_limit():
    import pytest
    from fastapi import HTTPException

    limit = settings.rate_limit_max_requests
    for _ in range(limit):
        limits.rate_limit(_Req("1.2.3.4"))  # all allowed
    with pytest.raises(HTTPException) as exc:
        limits.rate_limit(_Req("1.2.3.4"))  # one too many
    assert exc.value.status_code == 429


def test_separate_ips_are_independent():
    limits.rate_limit(_Req("10.0.0.1"))
    limits.rate_limit(_Req("10.0.0.2"))  # different IP — not blocked
    assert len(limits._hits["10.0.0.1"]) == 1
    assert len(limits._hits["10.0.0.2"]) == 1
