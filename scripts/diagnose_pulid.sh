#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${1:-/workspace/runpod-slim/ComfyUI}"
PY="$COMFY_ROOT/.venv-cu128/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON_BIN:-python3}"
fi

echo "[INFO] COMFY_ROOT=$COMFY_ROOT"
echo "[INFO] PYTHON=$PY"

echo
echo "=== 1) Process Check ==="
ps aux | grep -E "python.*main.py" | grep -v grep || true

echo
echo "=== 2) PuLID Node Directory ==="
if [[ -d "$COMFY_ROOT/custom_nodes/PuLID_ComfyUI" ]]; then
  ls -la "$COMFY_ROOT/custom_nodes/PuLID_ComfyUI" | sed -n '1,80p'
else
  echo "[FAIL] Missing: $COMFY_ROOT/custom_nodes/PuLID_ComfyUI"
fi

if [[ -d "$COMFY_ROOT/custom_nodes/PuLID_ComfyUI/.git" ]]; then
  echo "[OK] PuLID_ComfyUI is a git repo"
else
  echo "[WARN] PuLID_ComfyUI exists but is not a git repo"
fi

echo
echo "=== 3) Python Import Check ==="
"$PY" - <<'PY'
mods = ["facexlib", "insightface", "onnxruntime", "timm", "ftfy", "cv2", "torch", "torchvision"]
for m in mods:
    try:
        __import__(m)
        print("OK ", m)
    except Exception as e:
        print("FAIL", m, "=>", repr(e))
PY

echo
echo "=== 4) PuLID __init__ Load Test ==="
"$PY" - <<'PY'
import runpy
import traceback
p = "/workspace/runpod-slim/ComfyUI/custom_nodes/PuLID_ComfyUI/__init__.py"
try:
    runpy.run_path(p)
    print("OK  PuLID __init__ load")
except Exception:
    print("FAIL PuLID __init__ load")
    traceback.print_exc()
PY

echo
echo "=== 5) Recent Logs (if present) ==="
tail -n 200 "$COMFY_ROOT/comfyui.log" 2>/dev/null || true
tail -n 200 "$COMFY_ROOT/logs/comfyui.log" 2>/dev/null || true

echo
echo "=== 6) Targeted Error Scan ==="
if command -v rg >/dev/null 2>&1; then
  rg -n "PuLID|ModuleNotFoundError|ImportError|Traceback|Failed to import" "$COMFY_ROOT" \
    -g '!models/**' -g '!.git/**' -g '!.venv*/*' | tail -n 200 || true
else
  grep -RniE "PuLID|ModuleNotFoundError|ImportError|Traceback|Failed to import" "$COMFY_ROOT" \
    --exclude-dir=models --exclude-dir=.git --exclude-dir=.venv-cu128 --exclude-dir=__pycache__ \
    2>/dev/null | tail -n 200 || true
fi

echo
echo "[DONE] Diagnose completed."
