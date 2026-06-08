# Minimal vLLM server image (same approach as t3d-vllm-docker).
# pip install vllm resolves a compatible torch/CUDA stack — no NGC PyTorch / flash-attn pin fights.
# devel (not runtime): vLLM/FlashInfer JIT-compiles CUDA kernels at first start and needs nvcc.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install vllm

COPY cd-vllm/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7100

ENTRYPOINT ["/app/start.sh"]
