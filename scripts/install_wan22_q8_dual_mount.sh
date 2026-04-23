#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/workspace/runpod-slim/ComfyUI/models/checkpoints}"
KEY_FILE="${KEY_FILE:-/workspace/key.env}"
SOURCE_BASE="${SOURCE_BASE:-https://civitai.com/api/download/models}"
Q8H_VERSION_ID="${Q8H_VERSION_ID:-2540892}"
Q8L_VERSION_ID="${Q8L_VERSION_ID:-2540896}"
Q8H_NAME="${Q8H_NAME:-WAN2.2-NSFW-FastMove-V2-Q8H.gguf}"
Q8L_NAME="${Q8L_NAME:-WAN2.2-NSFW-FastMove-V2-Q8L.gguf}"

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

mkdir -p "$TARGET_DIR"

download_one() {
  local version_id="$1"
  local target_name="$2"
  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' RETURN

  python3 - "$version_id" "$target_name" "$TARGET_DIR" "$SOURCE_BASE" <<'PY'
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

version_id = sys.argv[1]
target_name = sys.argv[2]
target_dir = Path(sys.argv[3])
source_base = sys.argv[4].rstrip("/")
token = os.getenv("CIVITAI_TOKEN") or os.getenv("civitai")
if not token:
    raise SystemExit("missing Civitai token")

api_url = f"{source_base}/{version_id}"
req = Request(api_url, headers={"Authorization": f"Bearer {token}"})
with urlopen(req, timeout=120) as resp:
    if resp.status not in (200, 302, 303, 307, 308):
        raise SystemExit(f"unexpected Civitai status {resp.status}")
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^\";]+)"?', cd, re.I)
    filename = m.group(1).strip() if m else f"civitai_{version_id}.safetensors"
    suffix = Path(filename).suffix or ".safetensors"
    target = target_dir / (target_name if target_name.endswith(suffix) else f"{target_name}{suffix}")
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(target_dir), prefix=f".{target.name}.")
    try:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()
        os.replace(tmp.name, target)
        print(f"[saved] {target}")
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass
        raise
PY
}

download_one "$Q8H_VERSION_ID" "$Q8H_NAME"
download_one "$Q8L_VERSION_ID" "$Q8L_NAME"

echo "[done] $TARGET_DIR"
