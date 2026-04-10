#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_civitai_2110984.sh
#   COMFY_DIR=/workspace/runpod-slim/ComfyUI KEY_FILE=/workspace/key.env bash scripts/download_civitai_2110984.sh

COMFY_DIR="${COMFY_DIR:-/workspace/runpod-slim/ComfyUI}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
MODEL_VERSION_ID="${MODEL_VERSION_ID:-2110984}"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "ERROR: key file not found: $KEY_FILE" >&2
  echo "Hint: set KEY_FILE, e.g. KEY_FILE=/workspace/key.env" >&2
  exit 1
fi

if [[ ! -d "$COMFY_DIR" ]]; then
  echo "ERROR: ComfyUI path not found: $COMFY_DIR" >&2
  echo "Hint: set COMFY_DIR, e.g. COMFY_DIR=/workspace/runpod-slim/ComfyUI" >&2
  exit 1
fi

CIVITAI_KEY="$(sed -n 's/^civitai=//p' "$KEY_FILE" | head -n 1 | tr -d '\r\n')"
if [[ -z "${CIVITAI_KEY}" ]]; then
  echo "ERROR: missing civitai key in $KEY_FILE (expected: civitai=...)" >&2
  exit 1
fi

OUT_DIR="$COMFY_DIR/models/checkpoints"
mkdir -p "$OUT_DIR"

API_URL="https://civitai.com/api/download/models/${MODEL_VERSION_ID}"

# Ask Civitai for final filename so saved file matches model naming.
HEADERS="$(curl -sSI -H "Authorization: Bearer ${CIVITAI_KEY}" "$API_URL")"
FILENAME="$(
  printf '%s\n' "$HEADERS" \
    | awk 'BEGIN{IGNORECASE=1} /^content-disposition:/ {print}' \
    | sed -E 's/.*filename="?([^";]+).*/\1/' \
    | tr -d '\r\n'
)"

if [[ -z "${FILENAME}" ]]; then
  FILENAME="civitai_${MODEL_VERSION_ID}.safetensors"
fi

OUT_PATH="${OUT_DIR}/${FILENAME}"

echo "Downloading modelVersionId=${MODEL_VERSION_ID}"
echo "Target: ${OUT_PATH}"
curl -fL \
  -H "Authorization: Bearer ${CIVITAI_KEY}" \
  "$API_URL" \
  -o "$OUT_PATH"

echo "Done."
echo "Saved to: $OUT_PATH"
