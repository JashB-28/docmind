# 🧠 DocMind — Local RAG Chatbot

A fully local, privacy-first document Q&A system built with LangChain, Chroma, and Ollama. No API keys. No cloud costs. Your documents never leave your machine.

---

## Features

- **Multi-PDF support** — Upload and index multiple PDFs at once
- **Conversational memory** — Remembers context across follow-up questions
- **Confidence scores** — Every answer includes a retrieval confidence % per source
- **Page-level citations** — Know exactly which page of which document the answer came from
- **Document comparison** — Ask the same question across two documents side by side
- **Model selection** — Swap between Mistral, Llama 3, Phi-3, Gemma 2 live
- **Automated tests** — RAG pipeline validated with pytest + LLM-as-judge pattern

---

## Architecture

```
PDFs → PyPDFLoader → RecursiveTextSplitter → nomic-embed-text (Ollama)
                                                        ↓
                                              Chroma Vector Store
                                                        ↓
User Query → Embed Query → Similarity Search (k=5) → Prompt + Context
                                                        ↓
                                            Mistral / Llama3 (Ollama)
                                                        ↓
                                    Answer + Citations + Confidence Score
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Ollama (Mistral, Llama3, Phi3, Gemma2) |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | ChromaDB (local) |
| Orchestration | LangChain |
| UI | Streamlit |
| PDF loading | PyPDF + LangChain |
| Testing | pytest + LLM-as-judge |

---

## Quickstart

### 1. Install Ollama
Download from [ollama.com](https://ollama.com) and pull the required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your PDFs
```bash
mkdir data
# Copy your PDF files into the data/ folder
```

### 4. Index your documents
```bash
python populate_database.py
```

### 5. Run the app
```bash
streamlit run app.py
```

Or use the CLI directly:
```bash
python query_data.py "Your question here"
```

---

## Running Tests
```bash
pytest test_rag.py -v
```

Tests use an LLM-as-judge pattern — the model evaluates whether the RAG answer matches the expected response.

---

## How Confidence Scores Work

Chroma returns L2 distance scores for each retrieved chunk. DocMind converts these to a human-readable confidence percentage:

- **🟢 70%+** — High confidence, strong semantic match
- **🟡 40–69%** — Medium confidence, partial match
- **🔴 <40%** — Low confidence, weak match

---

## Project Structure

```
├── app.py                    # Streamlit UI
├── query_data.py             # RAG query pipeline with memory + citations
├── populate_database.py      # PDF ingestion + Chroma indexing
├── get_embedding_function.py # Embedding model config
├── test_rag.py               # Automated RAG tests
├── requirements.txt
└── data/                     # Drop your PDFs here
```

---

## Skills Demonstrated

`RAG` `LangChain` `ChromaDB` `Vector Embeddings` `Ollama` `Local LLMs` `Streamlit` `Prompt Engineering` `Conversational Memory` `Semantic Search` `pytest` `Python`
