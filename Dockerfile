FROM python:3.10-slim

RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_main.py ./app/main.py
COPY artifacts/  ./artifacts/

ENV MODEL_ID=Qwen/Qwen3-4B
ENV EMBED_ID=BAAI/bge-small-en-v1.5
ENV ARTIFACTS_DIR=/app/artifacts
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "app/main.py"]
