# Multimodal Fault Console — Inference API Dockerfile
# Deployed as Hugging Face Docker Space (CPU Basic, port 7860)
# or any container platform (set PORT env var).

FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire demo directory (includes artifacts, export, web/public/data)
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.34.3" \
    torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    numpy==1.24.3 \
    scikit-learn==1.6.1 \
    Pillow==10.4.0 \
    pydantic==2.11.4

# Expose port (HF Spaces uses 7860 by default)
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn demo.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
