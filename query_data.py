import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LCDocument
from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

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
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            return ChatOpenAI(model=model_name, api_key=key), "openai"
        else:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=model_name), "ollama"

    backend = os.getenv("LLM_BACKEND", "openai").lower()
    if backend == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL", "mistral")), "ollama"

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
    ), "openai"


def query_rag(query_text: str, model_name: str = None, chat_history: str = "", api_key: str = ""):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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
