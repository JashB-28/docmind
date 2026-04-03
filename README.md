# 🧠 DocMind — AI-Powered RAG Document Assistant

A production-ready document Q&A system built with LangChain, ChromaDB, and OpenAI. Upload PDFs, ask questions, and get answers with page-level citations and confidence scores. Supports both OpenAI cloud models and local Ollama models.

---

## Features

- **Multi-PDF support** — Upload and index multiple PDFs at once
- **Hybrid search** — Combines semantic vector search + BM25 keyword search for better retrieval
- **Conversational memory** — Remembers context across follow-up questions
- **Confidence scores** — Every answer includes a retrieval confidence % per source
- **Page-level citations** — Know exactly which page of which document the answer came from
- **Document comparison** — Ask the same question across two documents side by side
- **Summarization** — Generate structured summaries in multiple styles
- **Flexible model selection** — Use OpenAI (GPT-4o, GPT-4o-mini) or local Ollama models (Mistral, Llama3, etc.)
- **Dockerized** — Ready to deploy on Hugging Face Spaces or any cloud platform
- **Automated tests** — RAG pipeline validated with pytest + LLM-as-judge pattern

---

## Architecture

```
PDFs → PyPDFLoader → RecursiveTextSplitter → OpenAI text-embedding-3-small
                                                        ↓
                                              Chroma Vector Store
                                                        ↓
User Query → Embed Query → Hybrid Search (Vector + BM25) → Prompt + Context
                                                        ↓
                                        OpenAI GPT-4o-mini / Ollama LLM
                                                        ↓
                                    Answer + Citations + Confidence Score
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | OpenAI (GPT-4o, GPT-4o-mini) or Ollama (Mistral, Llama3, Phi3, Gemma2) |
| Embeddings | OpenAI text-embedding-3-small or nomic-embed-text via Ollama |
| Vector store | ChromaDB (local) |
| Keyword search | BM25 via rank_bm25 |
| Orchestration | LangChain |
| UI | Streamlit |
| PDF loading | PyPDF + LangChain |
| Testing | pytest + LLM-as-judge |
| Deployment | Docker + Hugging Face Spaces |

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/JashB-28/docmind.git
cd docmind
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
LLM_BACKEND=openai
EMBEDDING_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Only needed if switching to Ollama
OLLAMA_LLM_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### 4. Add your PDFs
```bash
mkdir data
# Copy your PDF files into the data/ folder
```

### 5. Index your documents
```bash
python populate_database.py
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## Switching to Local Models (Ollama)

To run fully locally without any API costs:

1. Install Ollama from [ollama.com](https://ollama.com) and pull models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

2. Update your `.env`:
```
LLM_BACKEND=ollama
EMBEDDING_BACKEND=ollama
```

3. Re-index your documents (embeddings are model-specific):
```bash
python populate_database.py --reset
```

---

## Docker Deployment

### Build and run locally
```bash
docker build -t docmind .
docker run -p 7860:7860 -e OPENAI_API_KEY=your-key docmind
```

### Deploy to Hugging Face Spaces
1. Create a new Space with **Docker** as the SDK
2. Push this repo to the Space
3. Add `OPENAI_API_KEY` in Space Settings → Secrets

---

## Running Tests
```bash
pytest test_rag.py -v
```

Tests use an LLM-as-judge pattern — OpenAI evaluates whether the RAG answer matches the expected response.

---

## How Confidence Scores Work

ChromaDB returns cosine similarity scores for each retrieved chunk. DocMind converts these to a human-readable confidence percentage:

- 🟢 **70%+** — High confidence, strong semantic match
- 🟡 **40–69%** — Medium confidence, partial match
- 🔴 **<40%** — Low confidence, weak match

---

## Project Structure

```
├── app.py                    # Streamlit UI
├── query_data.py             # Hybrid RAG pipeline with memory + citations
├── populate_database.py      # PDF ingestion + Chroma indexing
├── get_embedding_function.py # Embedding model config (OpenAI or Ollama)
├── test_rag.py               # Automated RAG evaluation tests
├── requirements.txt
├── Dockerfile
└── data/                     # Drop your PDFs here
```

---

## Skills Demonstrated

`RAG` `LangChain` `ChromaDB` `Hybrid Search` `BM25` `Vector Embeddings` `OpenAI API` `Ollama` `Local LLMs` `Streamlit` `Prompt Engineering` `Conversational Memory` `Semantic Search` `Docker` `pytest` `Python`
