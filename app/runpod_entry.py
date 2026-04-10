import os
import shlex
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import requests
import runpod

from handler import handler

COMFY_LOG_PATH = Path("/tmp/comfy.log")
COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/workspace/runpod-slim/ComfyUI"))
RUNPOD_VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_ROOT", "/runpod-volume"))


def _replace_with_symlink(link_path: Path, source_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        current = link_path.resolve(strict=False)
        if current == source_path.resolve(strict=False):
            return
        link_path.unlink()
    elif link_path.exists():
        backup = link_path.with_name(f"{link_path.name}.local.bak")
        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        link_path.rename(backup)
    link_path.symlink_to(source_path, target_is_directory=True)


def map_runpod_volume_if_present() -> None:
    # Serverless network volume is usually mounted at /runpod-volume.
    if not RUNPOD_VOLUME_ROOT.exists():
        return

    candidate_sources = [
        RUNPOD_VOLUME_ROOT / "ComfyUI" / "models",
        RUNPOD_VOLUME_ROOT / "runpod-slim" / "ComfyUI" / "models",
        RUNPOD_VOLUME_ROOT / "models",
    ]
    source_models = next((p for p in candidate_sources if p.exists() and p.is_dir()), None)
    if not source_models:
        print("[entry] /runpod-volume detected, but no models directory found; skip model linking")
        return

    target_models = COMFY_ROOT / "models"
    _replace_with_symlink(target_models, source_models)
    print(f"[entry] Linked models: {target_models} -> {source_models}")


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
    try:
        map_runpod_volume_if_present()
    except Exception:
        print("[entry] WARN: failed to map /runpod-volume, continue without linking")
        print(traceback.format_exc())
    start_comfy_if_needed()
    runpod.serverless.start({"handler": handler})
