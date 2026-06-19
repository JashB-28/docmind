# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend that also serves the built frontend ───────────────
FROM python:3.13-slim
WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STATIC_DIR=static

COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./
# Drop the compiled SPA where FastAPI serves it from (STATIC_DIR=static).
COPY --from=frontend /frontend/dist ./static

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
