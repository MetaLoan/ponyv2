import os
import shlex
import subprocess
import time
from pathlib import Path

import requests
import runpod

from handler import handler

COMFY_LOG_PATH = Path("/tmp/comfy.log")


def tail_comfy_log(lines: int = 120) -> str:
    if not COMFY_LOG_PATH.exists():
        return "<no /tmp/comfy.log found>"
    data = COMFY_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(data[-lines:]) if data else "<empty /tmp/comfy.log>"


def wait_comfy_ready(api_url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_url}/system_stats", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError(f"ComfyUI is not ready after {timeout_sec}s ({api_url})")


def start_comfy_if_needed() -> None:
    api_url = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188")
    boot_timeout = int(os.getenv("COMFY_BOOT_TIMEOUT", "360"))
    comfy_cmd = os.getenv(
        "COMFY_START_CMD",
        "python3 /workspace/runpod-slim/ComfyUI/main.py --listen 127.0.0.1 --port 8188",
    )

    try:
        r = requests.get(f"{api_url}/system_stats", timeout=2)
        if r.status_code == 200:
            print("[entry] ComfyUI already healthy, skip start")
            return
    except Exception:
        pass

    print(f"[entry] Starting ComfyUI: {comfy_cmd}")
    log = open(COMFY_LOG_PATH, "a", encoding="utf-8")
    proc = subprocess.Popen(
        shlex.split(comfy_cmd),
        stdout=log,
        stderr=log,
        cwd="/workspace/runpod-slim/ComfyUI",
    )

    try:
        wait_comfy_ready(api_url, boot_timeout)
    except Exception:
        log.flush()
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited early with code {proc.returncode}\n"
                f"---- /tmp/comfy.log (tail) ----\n{tail_comfy_log()}"
            )
        raise RuntimeError(
            f"ComfyUI did not become ready in time\n"
            f"---- /tmp/comfy.log (tail) ----\n{tail_comfy_log()}"
        )

    print("[entry] ComfyUI healthy")


if __name__ == "__main__":
    start_comfy_if_needed()
    runpod.serverless.start({"handler": handler})
