#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/runpod-slim/ComfyUI/models}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
HF_BASE_21="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files"
HF_BASE_22="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"

Q8H_NAME="${Q8H_NAME:-WAN2.2-NSFW-FastMove-V2-H.safetensors}"
Q8L_NAME="${Q8L_NAME:-WAN2.2-NSFW-FastMove-V2-L.safetensors}"
Q8H_URL="${Q8H_URL:-https://civitai.com/api/download/models/2477539}"
Q8L_URL="${Q8L_URL:-https://civitai.com/api/download/models/2477548}"
VAE_NAME="${VAE_NAME:-wan_2.1_vae.safetensors}"
VAE_URL="${VAE_URL:-$HF_BASE_21/vae/wan_2.1_vae.safetensors}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-umt5_xxl_fp8_e4m3fn_scaled.safetensors}"
TEXT_ENCODER_URL="${TEXT_ENCODER_URL:-$HF_BASE_21/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors}"
CLIP_VISION_NAME="${CLIP_VISION_NAME:-clip_vision_h.safetensors}"
CLIP_VISION_URL="${CLIP_VISION_URL:-$HF_BASE_21/clip_vision/clip_vision_h.safetensors}"

mkdir -p \
  "$MODELS_DIR/unet" \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/vae" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/clip_vision"

token="${CIVITAI_TOKEN:-${civitai:-}}"
if [[ -z "$token" && -f "$KEY_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$KEY_FILE"
  set +a
  token="${CIVITAI_TOKEN:-${civitai:-}}"
fi

download_http() {
  local url="$1"
  local dst="$2"
  if [[ -s "$dst" ]]; then
    echo "[skip] $dst"
    return 0
  fi
  echo "[download] $dst"
  curl -fL --retry 5 --retry-delay 2 --retry-connrefused \
    -A "Mozilla/5.0" \
    "$url" -o "$dst.tmp"
  mv -f "$dst.tmp" "$dst"
}

resolve_existing_model() {
  local name="$1"
  local primary_dir="$2"
  shift 2
  local candidates=("$primary_dir/$name" "$@")
  local path
  for path in "${candidates[@]}"; do
    if [[ -s "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

download_civitai() {
  local url="$1"
  local dst="$2"
  if [[ -s "$dst" ]]; then
    echo "[skip] $dst"
    return 0
  fi
  if [[ -z "$token" ]]; then
    echo "ERROR: missing CIVITAI_TOKEN/civitai for $dst" >&2
    exit 1
  fi
  echo "[download] $dst"
  curl -fL --retry 5 --retry-delay 2 --retry-connrefused \
    -A "Mozilla/5.0" \
    -H "Authorization: Bearer $token" \
    "$url" -o "$dst.tmp"
  mv -f "$dst.tmp" "$dst"
}

download_http "$VAE_URL" "$MODELS_DIR/vae/$VAE_NAME"
download_http "$TEXT_ENCODER_URL" "$MODELS_DIR/text_encoders/$TEXT_ENCODER_NAME"
download_http "$CLIP_VISION_URL" "$MODELS_DIR/clip_vision/$CLIP_VISION_NAME"

if existing_h="$(resolve_existing_model "$Q8H_NAME" "$MODELS_DIR/unet" "$MODELS_DIR/diffusion_models/$Q8H_NAME")"; then
  echo "[skip] $existing_h"
else
  download_civitai "$Q8H_URL" "$MODELS_DIR/unet/$Q8H_NAME"
fi

if existing_l="$(resolve_existing_model "$Q8L_NAME" "$MODELS_DIR/unet" "$MODELS_DIR/diffusion_models/$Q8L_NAME")"; then
  echo "[skip] $existing_l"
else
  download_civitai "$Q8L_URL" "$MODELS_DIR/unet/$Q8L_NAME"
fi

echo
echo "=== unet ==="
ls -lh "$MODELS_DIR/unet"
echo "=== diffusion_models ==="
ls -lh "$MODELS_DIR/diffusion_models"
echo "=== vae ==="
ls -lh "$MODELS_DIR/vae"
echo "=== text_encoders ==="
ls -lh "$MODELS_DIR/text_encoders"
echo "=== clip_vision ==="
ls -lh "$MODELS_DIR/clip_vision"
