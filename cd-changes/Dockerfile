# Inference client — calls vLLM over HTTP (see cd-vllm/).
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERENCE_BACKEND=vllm \
    VLLM_PORT=7100 \
    VLLM_START_SERVER=0 \
    VLLM_BASE_URL=http://127.0.0.1:7100/v1 \
    ENABLE_THINKING=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY cd-changes/requirements.txt /app/requirements.txt

RUN python -m pip install -U pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY cd-changes/ /app/

ENTRYPOINT ["python", "/app/qwen_family.py"]
