#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/civitai_to_s3.py"

TARGET_DIR="${TARGET_DIR:-/workspace/runpod-slim/ComfyUI/models/checkpoints}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
Q8H_URL="${Q8H_URL:-https://civitai.com/api/download/models/2540892}"
Q8L_URL="${Q8L_URL:-https://civitai.com/api/download/models/2540896}"
Q8H_NAME="${Q8H_NAME:-WAN2.2-NSFW-FastMove-V2-Q8H.gguf}"
Q8L_NAME="${Q8L_NAME:-WAN2.2-NSFW-FastMove-V2-Q8L.gguf}"

if [[ ! -f "$PY_SCRIPT" ]]; then
  curl -fsSL https://raw.githubusercontent.com/MetaLoan/ponyv2/main/scripts/civitai_to_s3.py -o "$PY_SCRIPT"
fi

mkdir -p "$TARGET_DIR"

if [[ -f "$KEY_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$KEY_FILE"
  set +a
fi

if [[ -z "${CIVITAI_TOKEN:-${civitai:-}}" ]]; then
  echo "ERROR: missing Civitai token. Set CIVITAI_TOKEN/civitai or point KEY_FILE to your key.env." >&2
  exit 1
fi

python3 "$PY_SCRIPT" "$Q8H_URL" --kind checkpoint --target-dir "$TARGET_DIR" --name "$Q8H_NAME" --key-file "$KEY_FILE"
python3 "$PY_SCRIPT" "$Q8L_URL" --kind checkpoint --target-dir "$TARGET_DIR" --name "$Q8L_NAME" --key-file "$KEY_FILE"

echo "[done] $TARGET_DIR"
