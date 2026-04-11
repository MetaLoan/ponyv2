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
WORKFLOW_API_PATH = Path(os.getenv("WORKFLOW_API_PATH", "/workspace/runpod-slim/ComfyUI/pulid_sdxl_workflow_web_api.json"))
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


def infer_mode(input_data: Dict) -> str:
    mode = str(input_data.get("mode", "")).strip()
    if mode:
        return mode
    if not input_data.get("reference_image") and not input_data.get("pose_image"):
        return "text_only"
    return "pose_then_face_swap" if input_data.get("pose_image") else "dual_pass_auto_pose"


def normalize_input(input_data: Dict) -> Dict:
    data = dict(input_data)
    data["mode"] = infer_mode(data)
    data.setdefault("width", 832)
    data.setdefault("height", 1216)
    data.setdefault("batch_size", 1)
    data.setdefault("base_steps", 8)
    data.setdefault("steps", 40)
    data.setdefault("cfg", 4)
    data.setdefault("base_cfg", 4)
    data.setdefault("base_denoise", 1)
    data.setdefault("denoise", 1)
    data.setdefault("base_sampler_name", "dpmpp_2m_sde")
    data.setdefault("base_scheduler", "karras")
    data.setdefault("sampler_name", "dpmpp_2m_sde")
    data.setdefault("scheduler", "karras")
    data.setdefault("pulid_method", "fidelity")
    data.setdefault("pulid_weight", 0.7)
    data.setdefault("pulid_start_at", 0.5)
    data.setdefault("pulid_end_at", 1.0)
    data.setdefault("cn_depth_strength", 0.6)
    data.setdefault("cn_depth_start_percent", 0.0)
    data.setdefault("cn_depth_end_percent", 1.0)
    data.setdefault("cn_pose_strength", 0.6)
    data.setdefault("cn_pose_start_percent", 0.0)
    data.setdefault("cn_pose_end_percent", 1.0)
    data.setdefault("jpg_quality", 85)
    if "enable_upscale" not in data and "use_upscale" in data:
        data["enable_upscale"] = bool(data.get("use_upscale"))
    data.setdefault("enable_upscale", False)
    if "enable_pulid" not in data:
        data["enable_pulid"] = data["mode"] not in {"pose_only", "text_only"}
    if "enable_lora" not in data:
        data["enable_lora"] = bool(data.get("loras"))
    return data


def validate_input(input_data: Dict) -> None:
    mode = input_data["mode"]
    if mode == "dual_pass_auto_pose":
        if "reference_image" not in input_data:
            raise RuntimeError("reference_image is required for dual_pass_auto_pose")
    elif mode == "pose_then_face_swap":
        if "reference_image" not in input_data:
            raise RuntimeError("reference_image is required for pose_then_face_swap")
        if "pose_image" not in input_data:
            raise RuntimeError("pose_image is required for pose_then_face_swap")
    elif mode == "pose_only":
        if "pose_image" not in input_data:
            raise RuntimeError("pose_image is required for pose_only")
    elif mode == "text_only":
        pass
    else:
        raise RuntimeError(f"Unsupported mode: {mode}")
    if not str(input_data.get("prompt", "")).strip():
        raise RuntimeError("prompt is required")


def _max_node_id(prompt: Dict) -> int:
    ids = [int(k) for k in prompt.keys() if str(k).isdigit()]
    return max(ids) if ids else 0


def apply_lora_chain(prompt: Dict, loras: List[Dict]) -> Tuple[List, List]:
    model_source = ["1", 0]
    clip_source = ["1", 1]
    if not loras:
        return model_source, clip_source

    max_id = _max_node_id(prompt)
    for idx, lora in enumerate(loras):
        node_id = "17" if idx == 0 else str(max_id + idx)
        if idx > 0:
            prompt[node_id] = {
                "inputs": {},
                "class_type": "LoraLoader",
                "_meta": {"title": f"Load LoRA {idx + 1}"},
            }
        strength_model = lora.get("strength_model", lora.get("strength", 0.0))
        strength_clip = lora.get("strength_clip", lora.get("strength", 0.0))
        prompt[node_id]["inputs"]["lora_name"] = lora["name"]
        prompt[node_id]["inputs"]["strength_model"] = float(strength_model)
        prompt[node_id]["inputs"]["strength_clip"] = float(strength_clip)
        prompt[node_id]["inputs"]["model"] = model_source
        prompt[node_id]["inputs"]["clip"] = clip_source
        model_source = [node_id, 0]
        clip_source = [node_id, 1]
    return model_source, clip_source


def prune_nodes(prompt: Dict, node_ids: Tuple[str, ...]) -> None:
    for node_id in node_ids:
        prompt.pop(node_id, None)


def apply_mode(prompt: Dict, input_data: Dict) -> Tuple[bool, bool]:
    mode = input_data["mode"]
    enable_pulid = bool(input_data.get("enable_pulid", mode not in {"pose_only", "text_only"}))
    uses_external_pose = False
    if mode == "dual_pass_auto_pose":
        set_pose_branch(prompt, False)
        prune_nodes(prompt, ("9",))
    elif mode in {"pose_then_face_swap", "pose_only"}:
        set_pose_branch(prompt, True)
        uses_external_pose = True
        prune_nodes(prompt, ("22", "23", "27"))
    elif mode == "text_only":
        prompt["14"]["inputs"]["positive"] = ["2", 0]
        prompt["14"]["inputs"]["negative"] = ["3", 0]
        prune_nodes(prompt, ("4", "5", "6", "7", "8", "9", "10", "11", "12", "22", "23", "24", "25", "26", "27", "28", "29"))
    return uses_external_pose, enable_pulid


def apply_overrides(prompt: Dict, input_data: Dict) -> Tuple[bool, bool]:
    if "prompt" in input_data:
        prompt["2"]["inputs"]["text"] = input_data["prompt"]
    if "negative_prompt" in input_data:
        prompt["3"]["inputs"]["text"] = input_data["negative_prompt"]

    if "width" in input_data:
        prompt["13"]["inputs"]["width"] = int(input_data["width"])
    if "height" in input_data:
        prompt["13"]["inputs"]["height"] = int(input_data["height"])
    if "batch_size" in input_data:
        prompt["13"]["inputs"]["batch_size"] = int(input_data["batch_size"])
    if "ckpt_name" in input_data:
        prompt["1"]["inputs"]["ckpt_name"] = input_data["ckpt_name"]

    if "base_steps" in input_data:
        prompt["22"]["inputs"]["steps"] = int(input_data["base_steps"])
    if "base_seed" in input_data:
        prompt["22"]["inputs"]["seed"] = int(input_data["base_seed"])
    if "base_cfg" in input_data:
        prompt["22"]["inputs"]["cfg"] = float(input_data["base_cfg"])
    if "base_sampler_name" in input_data:
        prompt["22"]["inputs"]["sampler_name"] = input_data["base_sampler_name"]
    if "base_scheduler" in input_data:
        prompt["22"]["inputs"]["scheduler"] = input_data["base_scheduler"]
    if "base_denoise" in input_data:
        prompt["22"]["inputs"]["denoise"] = float(input_data["base_denoise"])

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
    if "denoise" in input_data:
        prompt["14"]["inputs"]["denoise"] = float(input_data["denoise"])

    if "cn_depth_strength" in input_data:
        prompt["12"]["inputs"]["strength"] = float(input_data["cn_depth_strength"])
    if "cn_depth_start_percent" in input_data:
        prompt["12"]["inputs"]["start_percent"] = float(input_data["cn_depth_start_percent"])
    if "cn_depth_end_percent" in input_data:
        prompt["12"]["inputs"]["end_percent"] = float(input_data["cn_depth_end_percent"])
    if "cn_pose_strength" in input_data:
        prompt["26"]["inputs"]["strength"] = float(input_data["cn_pose_strength"])
    if "cn_pose_start_percent" in input_data:
        prompt["26"]["inputs"]["start_percent"] = float(input_data["cn_pose_start_percent"])
    if "cn_pose_end_percent" in input_data:
        prompt["26"]["inputs"]["end_percent"] = float(input_data["cn_pose_end_percent"])

    if "pulid_weight" in input_data:
        prompt["8"]["inputs"]["weight"] = float(input_data["pulid_weight"])
    if "pulid_start_at" in input_data:
        prompt["8"]["inputs"]["start_at"] = float(input_data["pulid_start_at"])
    if "pulid_end_at" in input_data:
        prompt["8"]["inputs"]["end_at"] = float(input_data["pulid_end_at"])
    if "pulid_method" in input_data:
        prompt["8"]["inputs"]["method"] = input_data["pulid_method"]

    enable_lora = bool(input_data.get("enable_lora"))
    model_source = ["1", 0]
    clip_source = ["1", 1]
    loras = input_data.get("loras")
    if enable_lora and isinstance(loras, list) and loras:
        valid_loras = [x for x in loras if isinstance(x, dict) and x.get("name")]
        model_source, clip_source = apply_lora_chain(prompt, valid_loras)

    prompt["2"]["inputs"]["clip"] = clip_source
    prompt["3"]["inputs"]["clip"] = clip_source
    if "22" in prompt:
        prompt["22"]["inputs"]["model"] = model_source
    if "8" in prompt:
        prompt["8"]["inputs"]["model"] = model_source
    enable_pulid = bool(input_data.get("enable_pulid", True))
    prompt["14"]["inputs"]["model"] = ["8", 0] if enable_pulid else model_source
    if not enable_lora:
        prune_nodes(prompt, ("17",))
    if not enable_pulid:
        prune_nodes(prompt, ("4", "5", "6", "7", "8"))

    use_upscale = bool(input_data.get("enable_upscale", input_data.get("use_upscale", False)))
    if "upscale_model_name" in input_data:
        prompt["18"]["inputs"]["model_name"] = input_data["upscale_model_name"]
    prompt["16"]["inputs"]["images"] = ["19", 0] if use_upscale else ["15", 0]
    if not use_upscale:
        prune_nodes(prompt, ("18", "19"))
    if not input_data.get("keep_intermediate", KEEP_INTERMEDIATE_DEFAULT):
        prune_nodes(prompt, ("27", "28", "29"))

    return use_upscale, enable_pulid


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
    endpoint_url = os.getenv("R2_ENDPOINT", "").strip()
    bucket = os.getenv("R2_BUCKET", os.getenv("Bucket", ""))
    public_url = os.getenv("R2_PUBLIC_URL", os.getenv("PublicURL", ""))
    region = os.getenv("R2_REGION", "auto")
    prefix = os.getenv("R2_PREFIX", "outputs").strip("/")

    if endpoint_url:
        endpoint_url = endpoint_url.rstrip("/")
    elif account_id:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    if not (access_key and secret_key and endpoint_url and bucket and public_url):
        return None, None

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
    data = normalize_input(event.get("input", {}) if isinstance(event, dict) else {})
    request_id = data.get("request_id") or uuid.uuid4().hex
    validate_input(data)

    prompt = load_json(WORKFLOW_API_PATH)
    v3 = load_json(WORKFLOW_V3_PATH)
    apply_v3_defaults(prompt, v3)

    ref_value = data.get("reference_image")
    if ref_value:
        ref_filename = resolve_media_to_comfy_filename(ref_value, "reference")
        prompt["7"]["inputs"]["image"] = ref_filename

    has_pose, enable_pulid = apply_mode(prompt, data)
    if has_pose:
        pose_filename = resolve_media_to_comfy_filename(data["pose_image"], "pose")
        prompt["9"]["inputs"]["image"] = pose_filename

    use_upscale, enable_pulid = apply_overrides(prompt, data)
    keep_intermediate = bool(data.get("keep_intermediate", KEEP_INTERMEDIATE_DEFAULT))
    jpg_quality = int(data.get("jpg_quality", 85))

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
        final_raw = _convert_to_jpg_bytes(final_raw, quality=jpg_quality)
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
            "mode": data["mode"],
            "pose_mode": "external_pose" if has_pose else ("text_only" if data["mode"] == "text_only" else "dual_pass_auto_pose"),
            "enable_pulid": enable_pulid,
            "enable_lora": bool(data.get("enable_lora")),
            "use_upscale": use_upscale,
            "final_format": "jpg" if use_upscale else final_ext.lstrip("."),
            "jpg_quality": jpg_quality if use_upscale else None,
        },
    }
