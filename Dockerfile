# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/workspace/runpod-slim/ComfyUI \
    COMFY_API_URL=http://127.0.0.1:8188

WORKDIR /workspace/runpod-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ComfyUI runtime requires torch explicitly in most clean CUDA base images.
# Pin to CUDA 12.4 wheels to match this image.
RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision torchaudio

# ComfyUI base
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /workspace/runpod-slim/ComfyUI
RUN python3 -m pip install -r /workspace/runpod-slim/ComfyUI/requirements.txt

COPY requirements-serverless.txt /workspace/runpod-slim/requirements-serverless.txt
RUN python3 -m pip install -r /workspace/runpod-slim/requirements-serverless.txt

COPY app/ /workspace/runpod-slim/app/
COPY config/v16_models.yaml /workspace/runpod-slim/config/v16_models.yaml
COPY scripts/install_v16_models.sh /workspace/runpod-slim/scripts/install_v16_models.sh
COPY workflows/pulid_sdxl_workflow_v3.json /workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3.json
COPY workflows/pulid_sdxl_workflow_v3_api.json /workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3_api.json

RUN chmod +x /workspace/runpod-slim/scripts/install_v16_models.sh

# Full packaging mode:
# docker build --build-arg BUILD_FULL_PACKAGE=1 --build-arg CIVITAI_TOKEN=xxx ...
ARG BUILD_FULL_PACKAGE=0
ARG CIVITAI_TOKEN=
RUN if [ "$BUILD_FULL_PACKAGE" = "1" ]; then \
      COMFY_ROOT=/workspace/runpod-slim/ComfyUI CIVITAI_TOKEN="$CIVITAI_TOKEN" /workspace/runpod-slim/scripts/install_v16_models.sh /workspace/runpod-slim/ComfyUI ; \
    fi

ENV WORKFLOW_API_PATH=/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3_api.json \
    WORKFLOW_V3_PATH=/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3.json \
    KEEP_INTERMEDIATE_DEFAULT=1 \
    R2_PREFIX=outputs

WORKDIR /workspace/runpod-slim/app
CMD ["python3", "runpod_entry.py"]
