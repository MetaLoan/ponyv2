#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/workspace/runpod-slim/ComfyUI/models/checkpoints}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
Q8H_URL="${Q8H_URL:-https://civitai.com/api/download/models/2540892}"
Q8L_URL="${Q8L_URL:-https://civitai.com/api/download/models/2540896}"
Q8H_NAME="${Q8H_NAME:-WAN2.2-NSFW-FastMove-V2-Q8H.gguf}"
Q8L_NAME="${Q8L_NAME:-WAN2.2-NSFW-FastMove-V2-Q8L.gguf}"

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

mkdir -p "$TARGET_DIR"

download_one() {
  local url="$1"
  local out_name="$2"
  local tmp_file
  tmp_file="$(mktemp "$TARGET_DIR/.${out_name}.XXXXXX")"
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

  mv -f "$tmp_file" "$TARGET_DIR/$out_name"
  trap - RETURN
}

download_one "$Q8H_URL" "$Q8H_NAME"
download_one "$Q8L_URL" "$Q8L_NAME"

echo "[done] $TARGET_DIR"
