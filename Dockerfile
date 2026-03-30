FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy model (free tier default)
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# HuggingFace cache directory (mount as volume for persistence)
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    PORT=8005

RUN mkdir -p /app/hf_cache

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
