#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/civitai_to_s3.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: missing $PY_SCRIPT" >&2
  exit 1
fi

read -r -p "选择要下载模型的类型 [1=checkpoint, 2=LoRA]: " KIND_CHOICE
case "${KIND_CHOICE:-}" in
  1) KIND="checkpoint" ;;
  2) KIND="lora" ;;
  *)
    echo "ERROR: invalid choice. Use 1 or 2." >&2
    exit 1
    ;;
esac

read -r -p "请输入下载地址: " SOURCE_URL
if [[ -z "${SOURCE_URL:-}" ]]; then
  echo "ERROR: empty download URL." >&2
  exit 1
fi

KEY_FILE="${KEY_FILE:-$HOME/Desktop/sdxl2img/key.env}"
S3_KEY_FILE="${S3_KEY_FILE:-$HOME/Desktop/sdxl2img/s3-credentials.txt}"

exec python3 "$PY_SCRIPT" "$SOURCE_URL" --kind "$KIND" --key-file "$KEY_FILE" --s3-key-file "$S3_KEY_FILE"
