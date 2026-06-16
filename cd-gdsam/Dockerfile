FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

# CUDA 11.8 nvcc typically supports up to SM89. (SM90 often requires CUDA 12.x)
ARG TORCH_ARCH="7.5;8.0;8.6;8.9"

ARG GD_SAM_LINK="https://github.com/IDEA-Research/Grounded-SAM-2.git"
ARG GD_SAM_TAG="b7a9c29f196edff0eb54dbe14588d7ae5e3dde28"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential cmake ninja-build \
    gcc-10 g++-10 \
    ffmpeg \
    libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-10 \
    CXX=g++-10 \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}" \
    CUDA_HOME="/usr/local/cuda" \
    TZ="UTC"

ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN mkdir -p /home/ubuntu/workdir
WORKDIR /home/ubuntu/workdir
RUN git clone "${GD_SAM_LINK}" Grounded-SAM-2
WORKDIR /home/ubuntu/workdir/Grounded-SAM-2
RUN git checkout "${GD_SAM_TAG}"

RUN python -m pip install --no-cache-dir --no-build-isolation -e . \
 && cd grounding_dino && python -m pip install --no-cache-dir --no-build-isolation -e .

WORKDIR /home/ubuntu/workdir/Grounded-SAM-2/checkpoints
RUN wget -q -O sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

WORKDIR /home/ubuntu/workdir/Grounded-SAM-2/gdino_checkpoints
RUN wget -q -O groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

WORKDIR /app
COPY cd-gdsam/requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir opencv-python-headless pycocotools supervision "transformers==4.39.3"
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY cd-gdsam/ /app/

ENV PYTHONPATH="/app:/home/ubuntu/workdir/Grounded-SAM-2:${PYTHONPATH}"

ENTRYPOINT ["python", "gdsam.py"]