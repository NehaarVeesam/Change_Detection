FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cd-data-prep/requirements.txt /app/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
 && python -m pip install -r /app/requirements.txt \
 && python -m playwright install --with-deps chromium

COPY cd-data-prep/ /app/

ENTRYPOINT ["python", "threejs_patches.py"]
