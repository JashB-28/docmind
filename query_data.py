import os
import json
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LCDocument
from get_embedding_function import (
    get_chroma_path,
    get_embedding_function,
    get_ollama_base_url,
)

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context extracted from documents to answer the question.
If the answer is not in the context, say "I couldn't find that in the provided documents."
Do not make up information.

Conversation so far:
{history}

Context from documents:
{context}

Question: {question}

Answer:"""


def get_llm(model_name: str = None, api_key: str = ""):
    if model_name:
        if model_name.startswith("gpt"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, api_key=api_key), "openai"
        else:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=model_name, base_url=get_ollama_base_url()), "ollama"

    backend = os.getenv("LLM_BACKEND", "openai").lower()
    if backend == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=os.getenv("OLLAMA_LLM_MODEL", "mistral"),
            base_url=get_ollama_base_url(),
        ), "ollama"

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
    ), "openai"


def query_rag(
    query_text: str,
    model_name: str = None,
    chat_history: str = "",
    api_key: str = "",
    embedding_backend: str = "openai",
):
    embedding_function = get_embedding_function(backend=embedding_backend, api_key=api_key)
    db = Chroma(
        persist_directory=get_chroma_path(embedding_backend),
        embedding_function=embedding_function,
    )

    # --- Hybrid search: vector + BM25 (manual merge, no EnsembleRetriever) ---
    all_docs = db.get(include=["documents", "metadatas"])
    if not all_docs["documents"]:
        return {"answer": "No relevant documents found.", "sources": [], "raw_sources": []}

    doc_objects = [
        LCDocument(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    # Vector search (semantic)
    vector_results = db.similarity_search_with_score(query_text, k=8)
    vector_docs = [doc for doc, _ in vector_results]

    # BM25 search (keyword)
    bm25_retriever = BM25Retriever.from_documents(doc_objects)
    bm25_retriever.k = 8
    bm25_docs = bm25_retriever.invoke(query_text)

    # Merge and deduplicate — vector first, then BM25 additions
    seen_ids = set()
    unique_docs = []
    for doc in vector_docs + bm25_docs:
        cid = doc.metadata.get("id", doc.page_content[:50])
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_docs.append(doc)

    unique_docs = unique_docs[:10]

    if not unique_docs:
        return {"answer": "No relevant documents found.", "sources": [], "raw_sources": []}

    context_text = "\n\n---\n\n".join([doc.page_content for doc in unique_docs])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        history=chat_history or "No prior conversation.",
        context=context_text,
        question=query_text,
    )

    model, backend = get_llm(model_name=model_name, api_key=api_key)

    if backend == "openai":
        answer = model.invoke(prompt).content
    else:
        answer = model.invoke(prompt)

    # Confidence scores from vector search
    score_map = {
        doc.metadata.get("id", doc.page_content[:50]): score
        for doc, score in vector_results
    }

    sources = []
    seen = set()
    for doc in unique_docs:
        chunk_id = doc.metadata.get("id", "")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        score = score_map.get(chunk_id, 1.0)
        confidence = max(0, round((1 - min(score, 1.5) / 1.5) * 100))
        filename = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        excerpt = doc.page_content

        sources.append({
            "filename": filename,
            "page": page,
            "confidence": confidence,
            "chunk_id": chunk_id,
            "excerpt": excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
        })

    return {
        "answer": answer,
        "sources": sources,
        "raw_sources": [doc.metadata.get("id") for doc in unique_docs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--chat-history", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--embedding-backend", default="openai")
    args = parser.parse_args()

    result = query_rag(
        query_text=args.query_text,
        model_name=args.model_name or None,
        chat_history=args.chat_history,
        api_key=args.api_key,
        embedding_backend=args.embedding_backend,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
