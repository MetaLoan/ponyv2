#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/civitai_to_s3.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  curl -fsSL https://raw.githubusercontent.com/MetaLoan/ponyv2/main/scripts/civitai_to_s3.py -o "$PY_SCRIPT"
fi

SOURCE_URL="${SOURCE_URL:-${1:-}}"
if [[ -z "${SOURCE_URL:-}" ]]; then
  echo "ERROR: SOURCE_URL is required" >&2
  exit 1
fi

KIND="${KIND:-checkpoint}"
TARGET_DIR="${TARGET_DIR:-/workspace/runpod-slim/ComfyUI/models/checkpoints}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"

exec python3 "$PY_SCRIPT" \
  "$SOURCE_URL" \
  --kind "$KIND" \
  --target-dir "$TARGET_DIR" \
  --key-file "$KEY_FILE"
