# Multi-stage Dockerfile for Filewise Evidence-Based Knowledge Assistant
FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential needed for some C extensions if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, sample data, and evaluation suite
COPY app/ ./app/
COPY data/sample_documents/ ./data/sample_documents/
COPY evaluation/ ./evaluation/
COPY pytest.ini .

# Create persistent vector store directory
RUN mkdir -p /app/data/vector_store

# Non-root user for security
RUN useradd -m filewiseuser && chown -R filewiseuser:filewiseuser /app
USER filewiseuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
