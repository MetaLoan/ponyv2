#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${1:-/workspace/runpod-slim/ComfyUI}"
PY="${PYTHON_BIN:-$COMFY_ROOT/.venv-cu128/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON_BIN:-python3}"
fi

echo "=== Environment ==="
echo "COMFY_ROOT=$COMFY_ROOT"
echo "PYTHON=$PY"
echo

echo "=== Python ==="
"$PY" -V
echo

echo "=== Core Python Packages ==="
"$PY" -m pip list --format=freeze | \
  egrep '^(numpy|scipy|onnx|onnxruntime|onnxruntime-gpu|insightface|torch|torchvision|torchaudio|facexlib|ftfy|timm|runpod|boto3|requests|Pillow|opencv-python|opencv-python-headless|safetensors|pyyaml|accelerate|diffusers|transformers|einops|psutil|pydantic|websocket-client|aiohttp|yarl|multidict|urllib3|typing-extensions|six|packaging|sympy|filelock|networkx|triton)==|^comfyui|^comfy' || true
echo

echo "=== ComfyUI Repo ==="
if [[ -d "$COMFY_ROOT/.git" ]]; then
  git -C "$COMFY_ROOT" rev-parse --short HEAD
  git -C "$COMFY_ROOT" branch --show-current || true
  git -C "$COMFY_ROOT" remote -v || true
else
  echo "[WARN] $COMFY_ROOT is not a git repo"
fi
echo

echo "=== Custom Nodes ==="
if [[ -d "$COMFY_ROOT/custom_nodes" ]]; then
  for d in "$COMFY_ROOT"/custom_nodes/*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    if [[ -d "$d/.git" ]]; then
      url="$(git -C "$d" remote get-url origin 2>/dev/null || echo unknown)"
      branch="$(git -C "$d" branch --show-current 2>/dev/null || echo unknown)"
      rev="$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo unknown)"
      printf '%s\t%s\t%s\t%s\n' "$name" "$branch" "$rev" "$url"
    else
      printf '%s\t%s\n' "$name" "no-git"
    fi
  done
else
  echo "[WARN] missing custom_nodes directory: $COMFY_ROOT/custom_nodes"
fi
