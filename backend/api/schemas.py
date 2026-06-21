"""Request and response models for the API."""

from pydantic import BaseModel, Field


class Source(BaseModel):
    filename: str
    page: int | str
    confidence: int
    chunk_id: str
    excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str
    provider: str = "openai"  # "openai" | "ollama"
    model: str | None = None
    api_key: str = ""
    history: str = ""
    filename: str | None = None  # restrict retrieval to one document


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str
    doc_a: str
    doc_b: str
    provider: str = "openai"
    model: str | None = None
    api_key: str = ""


class CompareAnswer(BaseModel):
    filename: str
    answer: str
    sources: list[Source]


class CompareResponse(BaseModel):
    results: list[CompareAnswer]


class IngestResponse(BaseModel):
    session_id: str
    indexed_files: list[str]
    total_chunks: int


class DocumentsResponse(BaseModel):
    session_id: str
    documents: list[str]


class HealthResponse(BaseModel):
    status: str
    pinecone_configured: bool
    default_llm_backend: str
    active_sessions: int
    default_provider: str
    enable_ollama: bool
    enable_s3: bool
