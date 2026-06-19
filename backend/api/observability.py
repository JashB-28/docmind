"""Observability: structured logging, request tracing, and LLM tracing.

Two layers, both safe to run with zero external services:

1. Always on — JSON logs to stdout with a per-request id and timing. On EC2 these
   flow straight to journald / CloudWatch, so you can see every request and every
   query's latency without signing up for anything.
2. Optional — a Langfuse callback handler (enabled by setting LANGFUSE_* keys)
   that records each LLM call as a trace you can inspect in the Langfuse UI:
   prompt, completion, token counts, latency, cost.
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar

from rag.config import settings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Carries the current request id into any log record emitted while handling it.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        # Merge structured fields passed via logger.info(..., extra={"extra": {...}}).
        if isinstance(getattr(record, "extra", None), dict):
            payload.update(record.extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # Quiet chatty third-party loggers so our structured lines stay readable.
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    """Emit a structured log line with arbitrary key/value fields."""
    logger.info(msg, extra={"extra": fields})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign each request an id and log its method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        logger = get_logger("api.request")
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            log_event(
                logger,
                "request",
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            request_id_var.reset(token)


_langfuse_handler = None
_langfuse_resolved = False


def get_langfuse_handler():
    """Return a Langfuse LangChain callback handler, or None when disabled.

    Resolved once and cached. Any failure degrades to None so tracing never
    breaks a request.
    """
    global _langfuse_handler, _langfuse_resolved
    if _langfuse_resolved:
        return _langfuse_handler
    _langfuse_resolved = True

    if not (settings.langfuse_public_key.strip() and settings.langfuse_secret_key.strip()):
        return None
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        _langfuse_handler = CallbackHandler()
        get_logger("api.observability").info("Langfuse tracing enabled")
    except Exception as exc:  # noqa: BLE001
        get_logger("api.observability").warning(f"Langfuse disabled: {exc}")
        _langfuse_handler = None
    return _langfuse_handler
