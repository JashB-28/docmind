import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from rag.config import settings
from rag.query import query_rag, stream_rag

from api.observability import get_langfuse_handler
from api.schemas import CompareAnswer, CompareRequest, CompareResponse, QueryRequest
from api.sessions import sessions

router = APIRouter(tags=["query"])


def _callbacks():
    handler = get_langfuse_handler()
    return [handler] if handler else None


def _resolve_model(provider: str, model: str | None) -> str:
    if provider == "ollama":
        return model or settings.ollama_llm_model
    if provider == "bedrock":
        return model or settings.bedrock_llm_model
    return model or settings.openai_llm_model


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/query")
def query(req: QueryRequest) -> StreamingResponse:
    """Stream an answer as Server-Sent Events: sources, then tokens, then done."""
    if not settings.pinecone_api_key.strip():
        raise HTTPException(503, "Server is missing PINECONE_API_KEY.")

    session = sessions.get(req.session_id)
    corpus = session.corpus if session else []
    model_name = _resolve_model(req.provider, req.model)
    callbacks = _callbacks()

    def event_stream():
        try:
            for kind, payload in stream_rag(
                req.question,
                model_name=model_name,
                chat_history=req.history,
                api_key=req.api_key,
                embedding_backend=req.provider,
                namespace=req.session_id,
                bm25_corpus=corpus,
                filename=req.filename,
                callbacks=callbacks,
            ):
                if kind == "sources":
                    yield _sse("sources", payload)
                elif kind == "token":
                    yield _sse("token", {"text": payload})
            yield _sse("done", {})
        except Exception as exc:  # noqa: BLE001 — report failures over the stream
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    if not settings.pinecone_api_key.strip():
        raise HTTPException(503, "Server is missing PINECONE_API_KEY.")

    session = sessions.get(req.session_id)
    corpus = session.corpus if session else []
    model_name = _resolve_model(req.provider, req.model)

    callbacks = _callbacks()
    results = []
    for filename in (req.doc_a, req.doc_b):
        result = query_rag(
            req.question,
            model_name=model_name,
            api_key=req.api_key,
            embedding_backend=req.provider,
            namespace=req.session_id,
            bm25_corpus=corpus,
            filename=filename,
            callbacks=callbacks,
        )
        results.append(
            CompareAnswer(
                filename=filename,
                answer=result["answer"],
                sources=result["sources"][:2],
            )
        )
    return CompareResponse(results=results)
