# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py .
COPY query_data.py .
COPY populate_database.py .
COPY get_embedding_function.py .

# ── Create data and chroma directories ───────────────────────────────────────
# data/   → users upload PDFs here at runtime
# chroma/ → vector DB persists here (mount a volume in production)
RUN mkdir -p data chroma

# ── Streamlit config ──────────────────────────────────────────────────────────
RUN mkdir -p /app/.streamlit
RUN echo '\
[server]\n\
port = 7860\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

# ── Expose port ───────────────────────────────────────────────────────────────
# Hugging Face Spaces requires port 7860
EXPOSE 7860

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
