#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${TARGET_ROOT:-/workspace/runpod-slim/ComfyUI/models}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
Q8H_URL="${Q8H_URL:-https://civitai.com/api/download/models/2376670}"
Q8L_URL="${Q8L_URL:-https://civitai.com/api/download/models/2376720}"
Q8H_NAME="${Q8H_NAME:-wan22I2V8StepsNSFWFP8_fp8Highnoise10.safetensors}"
Q8L_NAME="${Q8L_NAME:-wan22I2V8StepsNSFWFP8_fp8Lownoise10.safetensors}"
VAE_URL="${VAE_URL:-https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors}"
VAE_NAME="${VAE_NAME:-wan_2.1_vae.safetensors}"
CLIP_VISION_URL="${CLIP_VISION_URL:-https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors}"
CLIP_VISION_NAME="${CLIP_VISION_NAME:-clip_vision_h.safetensors}"
TEXT_ENCODER_URL="${TEXT_ENCODER_URL:-https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-umt5_xxl_fp8_e4m3fn_scaled.safetensors}"

token="${CIVITAI_TOKEN:-${civitai:-}}"
if [[ -z "$token" && -f "$KEY_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$KEY_FILE"
  set +a
  token="${CIVITAI_TOKEN:-${civitai:-}}"
fi

if [[ -z "$token" ]]; then
  echo "ERROR: missing Civitai token. Set CIVITAI_TOKEN/civitai or point KEY_FILE to your key.env." >&2
  exit 1
fi

download_one() {
  local url="$1"
  local out_name="$2"
  local target_dir="$3"
  local dst="$target_dir/$out_name"
  local tmp_file

  mkdir -p "$target_dir"

  if [[ -s "$dst" ]]; then
    echo "[SKIP] $dst already exists"
    return 0
  fi

  tmp_file="$(mktemp "$target_dir/.${out_name}.XXXXXX")"
  trap 'rm -f "$tmp_file"' RETURN

  echo "[download] $out_name"
  curl -fL --retry 5 --retry-delay 2 \
    -H "Authorization: Bearer $token" \
    -H "User-Agent: Mozilla/5.0" \
    "$url" -o "$tmp_file"

  if [[ ! -s "$tmp_file" ]]; then
    rm -f "$tmp_file"
    echo "ERROR: empty download for $out_name" >&2
    exit 1
  fi

  mv -f "$tmp_file" "$dst"
  trap - RETURN
}

download_one "$Q8H_URL" "$Q8H_NAME" "$TARGET_ROOT/unet"
download_one "$Q8L_URL" "$Q8L_NAME" "$TARGET_ROOT/unet"
download_one "$VAE_URL" "$VAE_NAME" "$TARGET_ROOT/vae"
download_one "$CLIP_VISION_URL" "$CLIP_VISION_NAME" "$TARGET_ROOT/clip_vision"
download_one "$TEXT_ENCODER_URL" "$TEXT_ENCODER_NAME" "$TARGET_ROOT/text_encoders"

echo "[done] $TARGET_ROOT"
