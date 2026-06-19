from fastapi import APIRouter

from api.schemas import HealthResponse
from api.sessions import sessions
from rag.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        pinecone_configured=bool(settings.pinecone_api_key.strip()),
        default_llm_backend=settings.llm_backend,
        active_sessions=len(sessions._sessions),
    )
