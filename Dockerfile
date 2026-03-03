# ─────────────────────────────────────────────────────────────────────────────
# Proactive Scam Intelligence — Inference API Container
#
# Builds a minimal, production-ready image for the FastAPI scoring endpoint.
#
# Build:
#   docker build -t scam-intelligence-api:latest .
#
# Run:
#   docker run -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     scam-intelligence-api:latest
#
# Interactive docs: http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Security: run as non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Install OS dependencies for PyTorch (CPU-only slim build)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (cache layer — only rebuilds if deps change)
COPY pyproject.toml .

# Install production dependencies only (no dev tools)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[dev]" --no-deps && \
    pip install --no-cache-dir \
        pandas pyarrow scikit-learn joblib \
        fastapi uvicorn[standard] pydantic \
        numpy

# Copy source code
COPY src/ src/
COPY models/ models/

# Do not copy data/ or reports/ into the image

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
