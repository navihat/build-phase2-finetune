# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /app

COPY src/serve/requirements.txt .

# torch CPU-only trước (image nhỏ hơn nhiều so với bản CUDA mặc định trên PyPI),
# rồi mới cài phần còn lại — tách riêng để lần cài sau không ghi đè nhầm bản GPU.
RUN pip install --no-cache-dir --timeout=120 --retries=5 --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir --timeout=120 --retries=5 --prefix=/install \
        -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Security: chạy bằng non-root user
RUN useradd -m appuser

COPY --chown=appuser:appuser src/serve/app.py ./src/serve/app.py
COPY --chown=appuser:appuser relation_classifier_v1 ./relation_classifier_v1

RUN mkdir -p /tmp/hf_home && chown appuser:appuser /tmp/hf_home

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/relation_classifier_v1 \
    HF_HOME=/tmp/hf_home \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER appuser

# Cloud Run set biến PORT lúc chạy (mặc định 8080) — không hard-code trong EXPOSE/CMD
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8080)}/health')" || exit 1

CMD exec uvicorn src.serve.app:app --host 0.0.0.0 --port ${PORT:-8080}
