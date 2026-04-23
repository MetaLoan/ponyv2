#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/runpod-slim/ComfyUI/models/clip_vision/EVA02_CLIP_L_336_psz14_s6B.pt}"

echo "MODEL_PATH=$MODEL_PATH"
python3 - "$MODEL_PATH" <<'PY'
import os
import sys
import torch

p = sys.argv[1]
print(f"SIZE={os.path.getsize(p)}")
with open(p, "rb") as f:
    head = f.read(32)
print(f"HEAD_HEX={head.hex()}")
print(f"HEAD_ASCII={repr(head)}")

x = torch.load(p, map_location="cpu")
print(f"TOP_TYPE={type(x).__name__}")
if isinstance(x, dict):
    keys = list(x.keys())
    print("TOP_KEYS=" + ",".join(str(k) for k in keys[:20]))
    sd = x.get("state_dict", x)
    if isinstance(sd, dict):
        sample = list(sd.keys())[:40]
        print("STATE_KEYS=" + ",".join(sample))
PY
