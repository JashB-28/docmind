import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from rag.config import settings
from rag.ingest import index_documents

from api.limits import rate_limit
from api.schemas import DocumentsResponse, IngestResponse
from api.sessions import sessions

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(rate_limit)])
async def ingest(
    files: list[UploadFile] = File(...),
    session_id: str = Form(""),
    provider: str = Form("openai"),
    api_key: str = Form(""),
) -> IngestResponse:
    if not settings.pinecone_api_key.strip():
        raise HTTPException(503, "Server is missing PINECONE_API_KEY.")
    if provider == "ollama" and not settings.enable_ollama:
        raise HTTPException(400, "The Ollama provider is disabled on this deployment.")
    if provider == "openai" and not (api_key or settings.openai_api_key):
        raise HTTPException(400, "An OpenAI API key is required for the OpenAI provider.")

    session = sessions.get_or_create(session_id or None)
    session.backend = provider
    max_bytes = settings.max_upload_mb * 1024 * 1024

    # Stage uploads in a throwaway temp dir; nothing is persisted on disk.
    with tempfile.TemporaryDirectory() as tmp_dir:
        names: list[str] = []
        for upload in files:
            if not upload.filename.lower().endswith(".pdf"):
                raise HTTPException(400, f"{upload.filename} is not a PDF.")
            data = await upload.read()
            if len(data) > max_bytes:
                raise HTTPException(
                    413, f"{upload.filename} exceeds the {settings.max_upload_mb} MB limit."
                )
            with open(os.path.join(tmp_dir, upload.filename), "wb") as out:
                out.write(data)
            names.append(upload.filename)

        try:
            result = index_documents(
                data_path=tmp_dir,
                backend=provider,
                api_key=api_key,
                namespace=session.session_id,
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # surface indexing errors to the client
            raise HTTPException(500, f"Indexing failed: {exc}") from exc

    # Cache the BM25 corpus in memory for this session and record filenames.
    session.corpus = result.chunks
    for name in names:
        if name not in session.documents:
            session.documents.append(name)

    return IngestResponse(
        session_id=session.session_id,
        indexed_files=names,
        total_chunks=result.total_chunks,
    )


@router.get("/{session_id}", response_model=DocumentsResponse)
def list_documents(session_id: str) -> DocumentsResponse:
    session = sessions.get(session_id)
    docs = session.documents if session else []
    return DocumentsResponse(session_id=session_id, documents=docs)


@router.delete("/{session_id}")
def clear_documents(session_id: str) -> dict:
    sessions.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
