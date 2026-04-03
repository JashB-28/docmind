from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from query_data import query_rag
from populate_database import load_documents, split_documents, add_to_chroma, clear_database

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background: #0f0f12; border-right: 1px solid #1e1e26; }
[data-testid="stSidebar"] * { color: #c9c9d4 !important; }

.sidebar-logo  { font-family: 'DM Serif Display', serif; font-size: 28px; color: #e8e4f0 !important; letter-spacing: -0.5px; padding: 8px 0 2px 0; }
.sidebar-sub   { font-size: 11px; color: #555566 !important; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 20px; }
.section-label { font-size: 10px; letter-spacing: 1.8px; text-transform: uppercase; color: #444455 !important; margin: 16px 0 6px 0; }
.doc-pill      { background: #1a1a24; border: 1px solid #2a2a38; border-radius: 6px; padding: 5px 10px; font-size: 12px; color: #9090a8 !important; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.main-header { font-family: 'DM Serif Display', serif; font-size: 42px; color: #e8e4f0; letter-spacing: -1px; line-height: 1; margin-bottom: 4px; }
.main-sub    { font-size: 14px; color: #888; margin-bottom: 28px; }

.msg-user      { background: #f0eeff; border-left: 3px solid #7c6fe0; border-radius: 0 12px 12px 0; padding: 12px 16px; margin: 8px 0; font-size: 15px; color: #2a2040; }
.msg-assistant { background: #f8f8fc; border-left: 3px solid #c4c0d8; border-radius: 0 12px 12px 0; padding: 12px 16px; margin: 8px 0; font-size: 15px; color: #1a1a2e; line-height: 1.7; }

.source-card     { background: #fafafa; border: 1px solid #eaeaf0; border-radius: 10px; padding: 10px 14px; margin: 6px 0; font-size: 13px; }
.source-filename { font-weight: 500; color: #3a3060; font-size: 13px; }
.source-page     { color: #888; font-size: 12px; }
.source-excerpt  { color: #555; font-size: 12px; margin-top: 8px; font-style: italic; line-height: 1.6; border-top: 1px solid #eee; padding-top: 8px; }
.conf-high { background: #e6f9ed; color: #1a7a40; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 500; }
.conf-mid  { background: #fff8e1; color: #8a6000; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 500; }
.conf-low  { background: #fdecea; color: #a02020; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 500; }

.summary-section { background: #f8f8fc; border: 1px solid #e2e0ef; border-radius: 12px; padding: 20px 24px; margin: 12px 0; line-height: 1.8; font-size: 15px; color: #1a1a2e; }
.summary-badge   { background: #ede9ff; color: #4a3aaa; border-radius: 8px; padding: 4px 12px; font-size: 12px; font-weight: 500; }

.compare-header { font-size: 13px; font-weight: 500; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 2px solid #eee; }

.empty-state { text-align: center; padding: 80px 20px; }
.empty-icon  { font-size: 48px; margin-bottom: 16px; }
.empty-title { font-size: 20px; font-weight: 500; color: #555; margin-bottom: 8px; }
.empty-sub   { font-size: 14px; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "docs_loaded": [],
    "mode": "chat",
    "model_name": "gpt-4o-mini",
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────────────────────────
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
OLLAMA_MODELS = {"mistral", "llama3", "llama3.2", "phi3", "gemma2"}

def is_openai_model(name: str) -> bool:
    return name in OPENAI_MODELS

def confidence_badge(conf: int) -> str:
    if conf >= 70:
        return f'<span class="conf-high">🟢 {conf}%</span>'
    elif conf >= 40:
        return f'<span class="conf-mid">🟡 {conf}%</span>'
    else:
        return f'<span class="conf-low">🔴 {conf}%</span>'

def render_sources(sources: list):
    for src in sources:
        badge = confidence_badge(src["confidence"])
        st.markdown(f"""
        <div class="source-card">
            <span class="source-filename">📄 {src['filename']}</span>
            <span class="source-page"> · Page {src['page']}</span>
            &nbsp;&nbsp;{badge}
            <div class="source-excerpt">{src['excerpt']}</div>
        </div>
        """, unsafe_allow_html=True)

def get_api_key() -> str:
    return st.session_state.openai_api_key or os.getenv("OPENAI_API_KEY", "")

def query_filtered(query: str, filename: str, model_name: str) -> dict:
    """RAG query filtered to a single document (Compare mode)."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document as LCDocument
    from langchain_community.retrievers import BM25Retriever
    from get_embedding_function import get_embedding_function

    CHROMA_PATH = "chroma"
    PROMPT = (
        "Answer based only on this context. If the information is not present, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Vector search filtered to this document
    vector_results = db.similarity_search_with_score(
        query, k=5, filter={"filename": filename}
    )
    vector_docs = [doc for doc, _ in vector_results]

    # BM25 on filtered docs only
    all_docs = db.get(include=["documents", "metadatas"])
    filtered_doc_objects = [
        LCDocument(page_content=text, metadata=meta)
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        if meta.get("filename") == filename
    ]

    if not filtered_doc_objects:
        return {"answer": "No relevant content found in this document.", "sources": []}

    bm25_retriever = BM25Retriever.from_documents(filtered_doc_objects)
    bm25_retriever.k = 5
    bm25_docs = bm25_retriever.invoke(query)

    # Merge and deduplicate
    seen_ids = set()
    unique_docs = []
    for doc in vector_docs + bm25_docs:
        cid = doc.metadata.get("id", doc.page_content[:50])
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_docs.append(doc)

    unique_docs = unique_docs[:8]

    if not unique_docs:
        return {"answer": "No relevant content found in this document.", "sources": []}

    context = "\n\n---\n\n".join([d.page_content for d in unique_docs])
    prompt = ChatPromptTemplate.from_template(PROMPT).format(context=context, question=query)

    if is_openai_model(model_name):
        from langchain_openai import ChatOpenAI
        answer = ChatOpenAI(model=model_name, api_key=get_api_key()).invoke(prompt).content
    else:
        from langchain_ollama import OllamaLLM
        answer = OllamaLLM(model=model_name).invoke(prompt)

    # Confidence from vector scores
    score_map = {
        doc.metadata.get("id", doc.page_content[:50]): score
        for doc, score in vector_results
    }

    sources = []
    seen = set()
    for doc in unique_docs:
        cid = doc.metadata.get("id", doc.page_content[:50])
        if cid in seen:
            continue
        seen.add(cid)
        score = score_map.get(cid, 1.0)
        confidence = max(0, round((1 - min(score, 1.5) / 1.5) * 100))
        excerpt = doc.page_content
        sources.append({
            "filename": doc.metadata.get("filename", filename),
            "page": doc.metadata.get("page_number", "?"),
            "confidence": confidence,
            "excerpt": excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
        })

    return {"answer": answer, "sources": sources}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">RAG · PDF Chat · Citations</div>', unsafe_allow_html=True)

    # API Key — always visible at the top
    st.markdown('<div class="section-label">OpenAI API Key</div>', unsafe_allow_html=True)
    key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=st.session_state.openai_api_key,
        label_visibility="collapsed",
        help="Your key is used only for this session and never stored on the server.",
    )
    st.session_state.openai_api_key = key_input
    if not key_input and not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Add your OpenAI API key to use GPT models.")

    # Documents
    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded:
        if st.button("⚡ Index Documents", use_container_width=True, type="primary"):
            os.makedirs("data", exist_ok=True)
            names = []
            for f in uploaded:
                with open(f"data/{f.name}", "wb") as out:
                    out.write(f.read())
                names.append(f.name)
            with st.spinner("Indexing — this may take a minute..."):
                try:
                    docs = load_documents()
                    chunks = split_documents(docs)
                    total = add_to_chroma(chunks)
                    st.session_state.docs_loaded = names
                    st.session_state.chat_history = []
                    st.success(f"✅ {total} chunks indexed from {len(names)} file(s)")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if st.session_state.docs_loaded:
        st.markdown('<div class="section-label">Loaded Files</div>', unsafe_allow_html=True)
        for name in st.session_state.docs_loaded:
            st.markdown(f'<div class="doc-pill">📄 {name}</div>', unsafe_allow_html=True)

    # Mode
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["💬 Chat", "⚖️ Compare Documents", "📋 Summarize"],
        label_visibility="collapsed",
    )
    st.session_state.mode = {"💬 Chat": "chat", "⚖️ Compare Documents": "compare", "📋 Summarize": "summarize"}[mode]

    # Model
    st.markdown("**OpenAI Models**")
    openai_model = st.selectbox(
        "OpenAI",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        label_visibility="collapsed",
        key="openai_models"
    )

    st.markdown("**Local Models (Ollama)**")
    ollama_model = st.selectbox(
        "Ollama",
        ["mistral", "llama3", "llama3.2", "phi3", "gemma2"],
        label_visibility="collapsed",
        key="ollama_models"
    )
    # Decide which one to use
    selected_model = openai_model if is_openai_model(openai_model) else ollama_model
    st.session_state.model_name = selected_model

    if not selected_model.startswith("---"):
        st.session_state.model_name = selected_model

    # Database & Chat controls
    st.markdown("---")
    st.markdown('<div class="section-label">Database</div>', unsafe_allow_html=True)
    if st.button("🗑 Clear Database", use_container_width=True):
        try:
            clear_database()
            st.session_state.docs_loaded = []
            st.session_state.chat_history = []
            st.success("✅ Database cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not clear database: {e}")

    st.markdown('<div class="section-label">Chat</div>', unsafe_allow_html=True)
    if st.button("🗑 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("LangChain · Chroma · OpenAI · Streamlit")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 DocMind</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Ask anything about your documents — with citations, confidence scores, and memory.</div>', unsafe_allow_html=True)

if not st.session_state.docs_loaded and st.session_state.mode != "summarize":
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No documents loaded yet</div>
        <div class="empty-sub">Upload PDFs in the sidebar and click <strong>Index Documents</strong> to begin.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Chat mode ─────────────────────────────────────────────────────────────────
if st.session_state.mode == "chat":
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} source(s)", expanded=False):
                    render_sources(msg["sources"])

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask", placeholder="Ask a question about your documents…", label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        history_str = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.chat_history[-6:]
        ])
        with st.spinner("Thinking..."):
            try:
                result = query_rag(
                    user_input,
                    model_name=st.session_state.model_name,
                    chat_history=history_str,
                    api_key=get_api_key(),
                )
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# ── Compare mode ──────────────────────────────────────────────────────────────
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Compare Documents")
    st.caption("Ask the same question and see how each document answers it side by side.")

    if len(st.session_state.docs_loaded) < 2:
        st.warning("Upload and index at least 2 documents to use Compare mode.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            doc_a = st.selectbox("Document A", st.session_state.docs_loaded, key="doc_a")
        with col_b:
            doc_b = st.selectbox(
                "Document B",
                st.session_state.docs_loaded,
                index=min(1, len(st.session_state.docs_loaded) - 1),
                key="doc_b",
            )

        compare_q = st.text_input("Question", placeholder="e.g. What are the main conclusions?")

        if st.button("⚖️ Compare", type="primary") and compare_q.strip():
            col1, col2 = st.columns(2)
            for col, doc in [(col1, doc_a), (col2, doc_b)]:
                with col:
                    st.markdown(f'<div class="compare-header">📄 {doc}</div>', unsafe_allow_html=True)
                    with st.spinner("Querying..."):
                        try:
                            r = query_filtered(compare_q, doc, st.session_state.model_name)
                            st.markdown(f'<div class="msg-assistant">{r["answer"]}</div>', unsafe_allow_html=True)
                            render_sources(r["sources"][:2])
                        except Exception as e:
                            st.error(str(e))


# ── Summarize mode ────────────────────────────────────────────────────────────
elif st.session_state.mode == "summarize":
    st.markdown("### 📋 Summarize Document")
    st.caption("Upload any PDF and get a structured summary. The file does not need to be indexed first.")

    sum_file = st.file_uploader("Upload a PDF to summarize", type=["pdf"], key="summarizer")

    SUMMARY_PROMPTS = {
        "Executive summary": (
            "Write a concise executive summary in 3–5 paragraphs. Cover the main purpose, "
            "key points, and conclusions. Use clear, professional language."
        ),
        "Key findings & conclusions": (
            "Extract and list the key findings, results, and conclusions as a numbered list. "
            "Be specific and include data or figures where available."
        ),
        "Section-by-section breakdown": (
            "Identify the main sections or chapters. For each, write 2–3 sentences summarizing "
            "its purpose and content. Use the section's own title as a heading."
        ),
        "Bullet point overview": (
            "Summarize the document as structured bullet points grouped by topic. "
            "Aim for completeness — cover all major points."
        ),
        "ELI5 — Plain language": (
            "Explain this document in very simple, everyday language that anyone can understand. "
            "Avoid jargon. Use short sentences and analogies where they help."
        ),
    }

    sum_type = st.selectbox("Summary style", list(SUMMARY_PROMPTS.keys()))

    if sum_file and st.button("📋 Generate Summary", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(sum_file.read())
            tmp_path = tmp.name

        with st.spinner("Reading and summarizing…"):
            try:
                pages = PyPDFLoader(tmp_path).load()
                text = "\n\n".join([p.page_content for p in pages[:20]])
                if len(text) > 12000:
                    text = text[:12000] + "\n\n[Document truncated for length]"

                prompt = ChatPromptTemplate.from_template(
                    "You are an expert document analyst.\n\n"
                    "Task: {instruction}\n\n"
                    "Document:\n{text}\n\n"
                    "Provide your response below, formatted clearly with headings or bullet points as appropriate:"
                ).format(instruction=SUMMARY_PROMPTS[sum_type], text=text)

                model_name = st.session_state.model_name
                if is_openai_model(model_name):
                    from langchain_openai import ChatOpenAI
                    summary = ChatOpenAI(model=model_name, api_key=get_api_key()).invoke(prompt).content
                else:
                    from langchain_ollama import OllamaLLM
                    summary = OllamaLLM(model=model_name).invoke(prompt)

                os.unlink(tmp_path)

                col1, col2, col3 = st.columns(3)
                col1.metric("Pages", len(pages))
                col2.metric("Words (approx)", f"{len(text.split()):,}")
                col3.metric("Model", model_name)

                st.markdown("---")
                st.markdown(
                    f'<span class="summary-badge">📄 {sum_type}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="summary-section">{summary}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Summarization failed: {e}")
