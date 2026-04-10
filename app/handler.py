import base64
import io
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import requests
from PIL import Image

COMFY_API_URL = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188")
COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/workspace/runpod-slim/ComfyUI"))
COMFY_INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", str(COMFY_ROOT / "input")))
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", str(COMFY_ROOT / "output")))
WORKFLOW_API_PATH = Path(os.getenv("WORKFLOW_API_PATH", "/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3_api.json"))
WORKFLOW_V3_PATH = Path(os.getenv("WORKFLOW_V3_PATH", "/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_v3.json"))
KEEP_INTERMEDIATE_DEFAULT = os.getenv("KEEP_INTERMEDIATE_DEFAULT", "1") == "1"
KEY_ENV_FILE = os.getenv("KEY_ENV_FILE", "")


def _load_key_env_file() -> None:
    if not KEY_ENV_FILE:
        return
    p = Path(KEY_ENV_FILE)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or k.startswith("#"):
            continue
        os.environ.setdefault(k, v)


def _guess_extension(content_type: str, fallback: str = ".jpg") -> str:
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    if not ext:
        return fallback
    return ".jpg" if ext == ".jpe" else ext


def _download_url_to_input(url: str, prefix: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    ext = _guess_extension(r.headers.get("content-type", ""))
    name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (COMFY_INPUT_DIR / name).write_bytes(r.content)
    return name


def _decode_base64_to_input(data: str, prefix: str) -> str:
    raw = data
    content_type = ""
    if data.startswith("data:") and "," in data:
        header, raw = data.split(",", 1)
        if ";base64" in header:
            content_type = header[5:].replace(";base64", "")
    blob = base64.b64decode(raw)
    ext = _guess_extension(content_type, ".jpg")
    name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (COMFY_INPUT_DIR / name).write_bytes(blob)
    return name


def resolve_media_to_comfy_filename(media: str, prefix: str) -> str:
    if media.startswith("http://") or media.startswith("https://"):
        return _download_url_to_input(media, prefix)
    return _decode_base64_to_input(media, prefix)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_v3_defaults(api_prompt: Dict, v3_workflow: Dict) -> None:
    nodes = {str(n["id"]): n for n in v3_workflow.get("nodes", [])}

    def widgets(nid: str) -> List:
        return nodes.get(nid, {}).get("widgets_values", [])

    w = widgets("17")
    if len(w) >= 3 and "17" in api_prompt:
        api_prompt["17"]["inputs"]["lora_name"] = w[0]
        api_prompt["17"]["inputs"]["strength_model"] = w[1]
        api_prompt["17"]["inputs"]["strength_clip"] = w[2]

    w = widgets("8")
    if len(w) >= 4 and "8" in api_prompt:
        api_prompt["8"]["inputs"]["method"] = w[0]
        api_prompt["8"]["inputs"]["weight"] = w[1]
        api_prompt["8"]["inputs"]["start_at"] = w[2]
        api_prompt["8"]["inputs"]["end_at"] = w[3]

    w = widgets("13")
    if len(w) >= 3 and "13" in api_prompt:
        api_prompt["13"]["inputs"]["width"] = w[0]
        api_prompt["13"]["inputs"]["height"] = w[1]
        api_prompt["13"]["inputs"]["batch_size"] = w[2]

    w = widgets("22")
    if len(w) >= 7 and "22" in api_prompt:
        api_prompt["22"]["inputs"]["seed"] = w[0]
        api_prompt["22"]["inputs"]["steps"] = w[2]
        api_prompt["22"]["inputs"]["cfg"] = w[3]
        api_prompt["22"]["inputs"]["sampler_name"] = w[4]
        api_prompt["22"]["inputs"]["scheduler"] = w[5]
        api_prompt["22"]["inputs"]["denoise"] = w[6]

    w = widgets("14")
    if len(w) >= 7 and "14" in api_prompt:
        api_prompt["14"]["inputs"]["seed"] = w[0]
        api_prompt["14"]["inputs"]["steps"] = w[2]
        api_prompt["14"]["inputs"]["cfg"] = w[3]
        api_prompt["14"]["inputs"]["sampler_name"] = w[4]
        api_prompt["14"]["inputs"]["scheduler"] = w[5]
        api_prompt["14"]["inputs"]["denoise"] = w[6]


def set_pose_branch(prompt: Dict, has_pose: bool) -> None:
    source = ["9", 0] if has_pose else ["23", 0]
    prompt["10"]["inputs"]["image"] = source
    prompt["24"]["inputs"]["image"] = source
    prompt["27"]["inputs"]["images"] = source


def apply_overrides(prompt: Dict, input_data: Dict) -> bool:
    if "prompt" in input_data:
        prompt["2"]["inputs"]["text"] = input_data["prompt"]
    if "negative_prompt" in input_data:
        prompt["3"]["inputs"]["text"] = input_data["negative_prompt"]

    if "width" in input_data:
        prompt["13"]["inputs"]["width"] = int(input_data["width"])
    if "height" in input_data:
        prompt["13"]["inputs"]["height"] = int(input_data["height"])

    if "base_steps" in input_data:
        prompt["22"]["inputs"]["steps"] = int(input_data["base_steps"])
    if "base_seed" in input_data:
        prompt["22"]["inputs"]["seed"] = int(input_data["base_seed"])
    if "base_sampler_name" in input_data:
        prompt["22"]["inputs"]["sampler_name"] = input_data["base_sampler_name"]
    if "base_scheduler" in input_data:
        prompt["22"]["inputs"]["scheduler"] = input_data["base_scheduler"]

    if "steps" in input_data:
        prompt["14"]["inputs"]["steps"] = int(input_data["steps"])
    if "cfg" in input_data:
        prompt["14"]["inputs"]["cfg"] = float(input_data["cfg"])
    if "seed" in input_data:
        prompt["14"]["inputs"]["seed"] = int(input_data["seed"])
    if "sampler_name" in input_data:
        prompt["14"]["inputs"]["sampler_name"] = input_data["sampler_name"]
    if "scheduler" in input_data:
        prompt["14"]["inputs"]["scheduler"] = input_data["scheduler"]

    if "cn_depth_strength" in input_data:
        prompt["12"]["inputs"]["strength"] = float(input_data["cn_depth_strength"])
    if "cn_pose_strength" in input_data:
        prompt["26"]["inputs"]["strength"] = float(input_data["cn_pose_strength"])

    if "pulid_weight" in input_data:
        prompt["8"]["inputs"]["weight"] = float(input_data["pulid_weight"])
    if "pulid_end_at" in input_data:
        prompt["8"]["inputs"]["end_at"] = float(input_data["pulid_end_at"])
    if "pulid_method" in input_data:
        prompt["8"]["inputs"]["method"] = input_data["pulid_method"]

    loras = input_data.get("loras")
    if isinstance(loras, list) and loras:
        first = loras[0]
        if "name" in first:
            prompt["17"]["inputs"]["lora_name"] = first["name"]
        if "strength" in first:
            s = float(first["strength"])
            prompt["17"]["inputs"]["strength_model"] = s
            prompt["17"]["inputs"]["strength_clip"] = s

    return bool(input_data.get("use_upscale", False))


def queue_prompt(prompt: Dict) -> str:
    payload = {"prompt": prompt, "client_id": str(uuid.uuid4())}
    r = requests.post(f"{COMFY_API_URL}/prompt", json=payload, timeout=60)
    if r.status_code >= 400:
        try:
            detail = json.dumps(r.json(), ensure_ascii=False)
        except Exception:
            detail = r.text[:4000]
        raise RuntimeError(f"Comfy /prompt rejected: status={r.status_code}, detail={detail}")
    body = r.json()
    if "prompt_id" not in body:
        raise RuntimeError(f"Comfy /prompt missing prompt_id: {json.dumps(body, ensure_ascii=False)}")
    return body["prompt_id"]


def get_object_info() -> Dict:
    r = requests.get(f"{COMFY_API_URL}/object_info", timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("Comfy /object_info returned unexpected payload")
    return data


def validate_required_node_types(prompt: Dict) -> None:
    object_info = get_object_info()
    available = set(object_info.keys())
    required = {
        str(node.get("class_type", "")).strip()
        for node in prompt.values()
        if isinstance(node, dict)
    }
    required.discard("")
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "Comfy missing required custom node types: "
            + ", ".join(missing)
            + ". Install/update corresponding custom_nodes and restart ComfyUI."
        )


def wait_history(prompt_id: str, timeout_sec: int = 1200) -> Dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        r = requests.get(f"{COMFY_API_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data and data[prompt_id].get("outputs"):
            return data[prompt_id]
        time.sleep(1.5)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def _read_output_image(image_desc: Dict) -> bytes:
    filename = image_desc["filename"]
    subfolder = image_desc.get("subfolder") or ""
    p = COMFY_OUTPUT_DIR / subfolder / filename if subfolder else COMFY_OUTPUT_DIR / filename
    return p.read_bytes()


def _convert_to_jpg_bytes(image_bytes: bytes, quality: int = 85) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def get_r2_client_and_config():
    access_key = os.getenv("R2_ACCESS_KEY", "")
    secret_key = os.getenv("R2_SECRET_KEY", "")
    account_id = os.getenv("R2_ACCOUNT_ID", "")
    bucket = os.getenv("R2_BUCKET", os.getenv("Bucket", ""))
    public_url = os.getenv("R2_PUBLIC_URL", os.getenv("PublicURL", ""))
    region = os.getenv("R2_REGION", "auto")
    prefix = os.getenv("R2_PREFIX", "outputs").strip("/")

    if not (access_key and secret_key and account_id and bucket and public_url):
        return None, None

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return s3, {"bucket": bucket, "public_url": public_url.rstrip("/"), "prefix": prefix}


def upload_bytes_to_r2(s3_client, cfg: Dict, key: str, data: bytes, content_type: str) -> str:
    s3_client.put_object(Bucket=cfg["bucket"], Key=key, Body=data, ContentType=content_type)
    return f"{cfg['public_url']}/{key}"


def collect_output_images(history_obj: Dict) -> Tuple[List[Dict], List[Dict]]:
    outputs = history_obj.get("outputs", {})
    final_images = outputs.get("16", {}).get("images", [])
    intermediate = []
    for nid in ("27", "28", "29"):
        intermediate.extend(outputs.get(nid, {}).get("images", []))
    return final_images, intermediate


def _summarize_history(history_obj: Dict) -> Dict:
    outputs = history_obj.get("outputs", {}) if isinstance(history_obj, dict) else {}
    output_nodes = sorted(outputs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    image_counts = {}
    for nid, out in outputs.items():
        imgs = out.get("images", []) if isinstance(out, dict) else []
        if imgs:
            image_counts[str(nid)] = len(imgs)

    status = history_obj.get("status", {}) if isinstance(history_obj, dict) else {}
    messages = status.get("messages", []) if isinstance(status, dict) else []
    error_messages = []
    for m in messages[-20:]:
        if isinstance(m, (list, tuple)) and len(m) >= 2:
            level = m[0]
            payload = m[1]
            if level == "execution_error":
                error_messages.append(payload)

    return {
        "output_nodes": output_nodes,
        "image_counts": image_counts,
        "status_str": status.get("status_str") if isinstance(status, dict) else None,
        "execution_errors": error_messages,
    }


def handler(event: Dict) -> Dict:
    _load_key_env_file()
    data = event.get("input", {}) if isinstance(event, dict) else {}
    request_id = data.get("request_id") or uuid.uuid4().hex

    if "reference_image" not in data:
        return {"ok": False, "error": "reference_image is required"}

    prompt = load_json(WORKFLOW_API_PATH)
    v3 = load_json(WORKFLOW_V3_PATH)
    apply_v3_defaults(prompt, v3)

    ref_filename = resolve_media_to_comfy_filename(data["reference_image"], "reference")
    prompt["7"]["inputs"]["image"] = ref_filename

    pose_value = data.get("pose_image")
    has_pose = bool(pose_value)
    if has_pose:
        pose_filename = resolve_media_to_comfy_filename(pose_value, "pose")
        prompt["9"]["inputs"]["image"] = pose_filename
    set_pose_branch(prompt, has_pose)

    use_upscale = apply_overrides(prompt, data)
    keep_intermediate = bool(data.get("keep_intermediate", KEEP_INTERMEDIATE_DEFAULT))

    validate_required_node_types(prompt)
    prompt_id = queue_prompt(prompt)
    history_obj = wait_history(prompt_id)
    final_images, intermediate_images = collect_output_images(history_obj)
    if not final_images:
        return {
            "ok": False,
            "error": "No final image produced by node 16",
            "prompt_id": prompt_id,
            "history_debug": _summarize_history(history_obj),
        }

    final_raw = _read_output_image(final_images[0])
    final_ext = Path(final_images[0]["filename"]).suffix.lower() or ".png"
    final_content_type = "image/png" if final_ext == ".png" else "image/jpeg"

    if use_upscale:
        final_raw = _convert_to_jpg_bytes(final_raw, quality=85)
        final_ext = ".jpg"
        final_content_type = "image/jpeg"

    s3, s3_cfg = get_r2_client_and_config()
    if not s3:
        return {
            "ok": True,
            "prompt_id": prompt_id,
            "request_id": request_id,
            "warning": "R2 env vars missing; returning local filenames only",
            "final_local_file": final_images[0]["filename"],
            "intermediate_local_files": [x["filename"] for x in intermediate_images],
        }

    final_key = f"{s3_cfg['prefix']}/{request_id}/final{final_ext}"
    final_url = upload_bytes_to_r2(s3, s3_cfg, final_key, final_raw, final_content_type)

    intermediate_urls: List[str] = []
    if keep_intermediate:
        for idx, img_desc in enumerate(intermediate_images, start=1):
            raw = _read_output_image(img_desc)
            ext = Path(img_desc["filename"]).suffix.lower() or ".png"
            content_type = "image/png" if ext == ".png" else "image/jpeg"
            key = f"{s3_cfg['prefix']}/{request_id}/intermediate/{idx:02d}_{img_desc['filename']}"
            intermediate_urls.append(upload_bytes_to_r2(s3, s3_cfg, key, raw, content_type))

    return {
        "ok": True,
        "prompt_id": prompt_id,
        "request_id": request_id,
        "storage": {"provider": "r2", "bucket": s3_cfg["bucket"]},
        "final_url": final_url,
        "intermediate_urls": intermediate_urls,
        "meta": {
            "pose_mode": "external_pose" if has_pose else "dual_pass_auto_pose",
            "use_upscale": use_upscale,
            "final_format": "jpg" if use_upscale else final_ext.lstrip("."),
            "jpg_quality": 85 if use_upscale else None,
        },
    }
