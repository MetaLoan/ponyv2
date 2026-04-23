#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/civitai_to_s3.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: missing $PY_SCRIPT" >&2
  exit 1
fi

SOURCE_URL="${SOURCE_URL:-${1:-}}"
if [[ -z "${SOURCE_URL:-}" ]]; then
  echo "ERROR: SOURCE_URL is required" >&2
  exit 1
fi

KIND="${KIND:-checkpoint}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
S3_KEY_FILE="${S3_KEY_FILE:-/workspace/s3-credentials.txt}"
MODEL_NAME="${MODEL_NAME:-}"

ARGS=(
  "$SOURCE_URL"
  --kind "$KIND"
  --download-dir "$DOWNLOAD_DIR"
  --key-file "$KEY_FILE"
  --s3-key-file "$S3_KEY_FILE"
)

if [[ -n "${MODEL_NAME:-}" ]]; then
  ARGS+=(--name "$MODEL_NAME")
fi

exec python3 "$PY_SCRIPT" "${ARGS[@]}"
