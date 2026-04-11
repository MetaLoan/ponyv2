# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    COMFY_ROOT=/workspace/runpod-slim/ComfyUI \
    COMFY_API_URL=http://127.0.0.1:8188

WORKDIR /workspace/runpod-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
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

# Required custom nodes for V16 workflow
RUN set -eux; \
    mkdir -p /workspace/runpod-slim/ComfyUI/custom_nodes; \
    for pair in \
      "https://github.com/Fannovel16/comfyui_controlnet_aux.git /workspace/runpod-slim/ComfyUI/custom_nodes/comfyui_controlnet_aux" \
      "https://github.com/cubiq/PuLID_ComfyUI.git /workspace/runpod-slim/ComfyUI/custom_nodes/PuLID_ComfyUI"; do \
      url="$(echo "$pair" | awk '{print $1}')"; \
      dst="$(echo "$pair" | awk '{print $2}')"; \
      ok=0; \
      for i in 1 2 3; do \
        git clone --depth 1 "$url" "$dst" && ok=1 && break || true; \
        rm -rf "$dst"; \
        sleep 5; \
      done; \
      [ "$ok" = "1" ]; \
    done

RUN python3 -m pip install --retries 5 --timeout 120 -r /workspace/runpod-slim/ComfyUI/custom_nodes/PuLID_ComfyUI/requirements.txt && \
    python3 -m pip install --retries 5 --timeout 120 -r /workspace/runpod-slim/ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt
# Global fallback deps for node coexistence.
RUN python3 -m pip install --retries 5 --timeout 120 --no-cache-dir --prefer-binary \
    "facexlib==0.3.0" \
    "ftfy==6.3.1" \
    "timm==1.0.26" \
    "huggingface_hub" \
    "onnxruntime-gpu==1.19.2"
RUN set -eux; \
    python3 -m pip uninstall -y numpy scipy onnx onnxruntime onnxruntime-gpu insightface || true; \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"; \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
      "scipy==1.11.4" \
      "onnx==1.18.0" \
      "onnxruntime-gpu==1.19.2" \
      "insightface==0.7.3"; \
    echo "[PIP] installed compatible runtime pins (numpy 1.26.4 / scipy 1.11.4 / onnx 1.18.0 / ort 1.19.2 / insightface 0.7.3)"

COPY requirements-serverless.txt /workspace/runpod-slim/requirements-serverless.txt
RUN python3 -m pip install -r /workspace/runpod-slim/requirements-serverless.txt

COPY app/ /workspace/runpod-slim/app/
COPY config/v16_models.yaml /workspace/runpod-slim/config/v16_models.yaml
COPY scripts/install_v16_models.sh /workspace/runpod-slim/scripts/install_v16_models.sh
COPY workflows/pulid_sdxl_workflow_v3.json /workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3.json
COPY workflows/pulid_sdxl_workflow_v3_api.json /workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3_api.json
COPY workflows/pulid_sdxl_workflow_web_api.json /workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_web_api.json

RUN chmod +x /workspace/runpod-slim/scripts/install_v16_models.sh

# Optional full-package mode:
# docker build --build-arg BUILD_FULL_PACKAGE=1 --build-arg CIVITAI_TOKEN=xxx ...
ARG BUILD_FULL_PACKAGE=0
ARG CIVITAI_TOKEN=
RUN if [ "$BUILD_FULL_PACKAGE" = "1" ]; then \
      COMFY_ROOT=/workspace/runpod-slim/ComfyUI CIVITAI_TOKEN="$CIVITAI_TOKEN" \
      /workspace/runpod-slim/scripts/install_v16_models.sh /workspace/runpod-slim/ComfyUI ; \
    fi

ENV WORKFLOW_API_PATH=/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_web_api.json \
    WORKFLOW_V3_PATH=/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3.json \
    KEEP_INTERMEDIATE_DEFAULT=1 \
    R2_PREFIX=outputs

WORKDIR /workspace/runpod-slim/app
CMD ["python3", "runpod_entry.py"]
