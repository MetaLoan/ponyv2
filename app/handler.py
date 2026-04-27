import base64
import io
import json
import fcntl
import mimetypes
import os
import subprocess
import tempfile
import time
import sys
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
MODEL_SYNC_ENABLED = os.getenv("MODEL_SYNC_ON_DEMAND", "1") == "1"
MODEL_SYNC_LOCK_PATH = Path(os.getenv("MODEL_SYNC_LOCK_PATH", "/tmp/ponyv2-model-sync.lock"))
MODEL_S3_ACCESS_KEY = os.getenv("MODEL_S3_ACCESS_KEY_ID", os.getenv("S3_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "")))
MODEL_S3_SECRET_KEY = os.getenv("MODEL_S3_SECRET_ACCESS_KEY", os.getenv("S3_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "")))
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET", os.getenv("S3_BUCKET", "")).strip()
MODEL_S3_ENDPOINT = os.getenv("MODEL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "")).strip()
MODEL_S3_REGION = os.getenv("MODEL_S3_REGION", os.getenv("S3_REGION", "eu-ro-1"))
MODEL_S3_ROOT_PREFIX = os.getenv("MODEL_S3_ROOT_PREFIX", os.getenv("S3_MODEL_ROOT_PREFIX", "runpod-slim/ComfyUI/models")).strip("/")

MODEL_KIND_TO_FOLDER = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "upscale_model": "upscale_models",
}

QWEN_API_URL = os.getenv(
    "DASHSCOPE_API_URL",
    os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    ),
)
QWEN_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-image-2.0-pro")
QWEN_DATA_INSPECTION = os.getenv(
    "DASHSCOPE_DATA_INSPECTION_HEADER",
    '{"input":"disable", "output": "disable"}',
)
QWEN_DEFAULT_SWAP_PROMPT = "以图1为最终画面底图，严格保留图1的人物姿势、构图、服装、光照、背景和沙滩环境；仅将图2中的面部特征自然融合到图1人物脸上，保持真实自然、五官清晰、肤质统一；图3如存在，仅作为辅助参考，不要改变其他区域。"
QWEN_DEFAULT_EDIT_PROMPT = "将图中的角色脸部特征形象进行调整，使其符合如下描述中关于脸部的特征描述:{{生图提示词的主提示词变量}}"
I2V_API_URL = os.getenv(
    "DASHSCOPE_I2V_API_URL",
    os.getenv(
        "DASHSCOPE_VIDEO_API_URL",
        "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
    ),
)
I2V_MODEL = os.getenv("DASHSCOPE_I2V_MODEL", "wan2.7-i2v")
I2V_DATA_INSPECTION = os.getenv("DASHSCOPE_DATA_INSPECTION_HEADER", QWEN_DATA_INSPECTION)
I2V_DEFAULT_PROMPT = "保持主体一致，生成自然流畅、画面连贯的动态视频，镜头稳定，动作真实，细节清晰。"
WAN_EXTEND_ANY_FRAME_MODE = "wan2_2_i2v_extend_any_frame"
WAN_EXTEND_ANY_FRAME_MODEL = os.getenv("WAN_EXTEND_ANY_FRAME_MODEL", "wan2.2-kf2v-flash")
WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT = 161
WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT = "沙滩，海边，晴天，自然光，蓝天白云，海浪，金色细沙，轻微海风，真实摄影感，画面通透，动作自然连贯，镜头稳定，细节清晰，电影感成片"
WAN_WORKFLOW_API_PATH = Path(os.getenv("WAN_WORKFLOW_API_PATH", "/workspace/runpod-slim/ComfyUI/wan2_2_i2v_extend_any_frame_api.json"))
WAN_EXECUTION_BACKEND = os.getenv("WAN_EXECUTION_BACKEND", "auto").strip().lower()
WAN_DEFAULT_WIDTH = int(os.getenv("WAN_DEFAULT_WIDTH", "480"))
WAN_DEFAULT_HEIGHT = int(os.getenv("WAN_DEFAULT_HEIGHT", "832"))
WAN_VIDEO_FPS = int(os.getenv("WAN_VIDEO_FPS", "16"))
WAN_VAE_NAME = os.getenv("WAN_VAE_NAME", "wan_2.1_vae.safetensors")
WAN_CLIP_VISION_NAME = os.getenv("WAN_CLIP_VISION_NAME", "clip_vision_h.safetensors")
WAN_CLIP_NAME = os.getenv("WAN_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
WAN_UNET_HIGH_NAME = os.getenv("WAN_UNET_HIGH_NAME", "WAN2.2-NSFW-FastMove-V2-H.safetensors")
WAN_UNET_LOW_NAME = os.getenv("WAN_UNET_LOW_NAME", "WAN2.2-NSFW-FastMove-V2-L.safetensors")


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


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


def _decode_media_bytes(media: str) -> Tuple[bytes, str]:
    if media.startswith("http://") or media.startswith("https://"):
        r = requests.get(media, timeout=60)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        return r.content, content_type
    raw = media
    content_type = ""
    if media.startswith("data:") and "," in media:
        header, raw = media.split(",", 1)
        content_type = header[5:].replace(";base64", "")
    blob = base64.b64decode(raw)
    return blob, content_type


def _fit_image_for_qwen(image: Image.Image, max_side: int = 2048, min_side: int = 512) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if min(w, h) < min_side:
        min_scale = min_side / min(w, h)
        if max(w, h) * min_scale <= max_side:
            scale = max(scale, min_scale)
    if abs(scale - 1.0) < 1e-6:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _media_to_qwen_data_url(media: str) -> Tuple[str, Tuple[int, int]]:
    blob, _ = _decode_media_bytes(media)
    img = Image.open(io.BytesIO(blob))
    img = _fit_image_for_qwen(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}", img.size


def _image_bytes_to_qwen_data_url(blob: bytes) -> Tuple[str, Tuple[int, int]]:
    img = Image.open(io.BytesIO(blob))
    img = _fit_image_for_qwen(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}", img.size


def _media_to_dashscope_accessible_url(media: str, request_id: str, prefix: str) -> str:
    media = (media or "").strip()
    if not media:
        raise RuntimeError(f"{prefix} media is required")
    if media.startswith("http://") or media.startswith("https://"):
        return media
    s3, s3_cfg = get_r2_client_and_config()
    if not s3:
        raise RuntimeError(
            f"{prefix} media must be a public http(s) URL when R2 is not configured"
        )
    blob, content_type = _decode_media_bytes(media)
    ext = _guess_extension(content_type or "image/png", ".png")
    key = f"dashscope_inputs/{request_id}/{prefix}_{uuid.uuid4().hex}{ext}"
    s3.put_object(Bucket=s3_cfg["bucket"], Key=key, Body=blob, ContentType=content_type or "image/png")
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": s3_cfg["bucket"], "Key": key},
        ExpiresIn=6 * 60 * 60,
    )


def _dashscope_text_chat(prompt: str, system: str = None, model: str = "qwen-max") -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return ""
    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    payload = {
        "model": model,
        "input": {"messages": []},
        "parameters": {"result_format": "message"},
    }
    if system:
        payload["input"]["messages"].append({"role": "system", "content": system})
    payload["input"]["messages"].append({"role": "user", "content": prompt})
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["output"]["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[dashscope] chat failed: {e}")
        return ""


def _generate_segment_prompts(main_prompt: str, segment_count: int) -> List[str]:
    if segment_count <= 1:
        return [main_prompt]
    system_msg = (
        "You are a video storyboard assistant. The user is generating a long video by splitting a main action into multiple segments. "
        "Your goal is to decompose the main prompt into a sequence of detailed sub-actions, one for each segment, to ensure a logical and coherent progression. "
        "For example, if the main prompt is 'a student putting on shoes', and there are 3 segments: "
        "Segment 1: 'a student putting on shoes', "
        "Segment 2: 'the student tying the left shoelace', "
        "Segment 3: 'the student tying the right shoelace'. "
        "Output ONLY the segment prompts, one per line. Do not include numbers, labels, or extra text. "
        "Ensure each segment prompt is descriptive and maintains consistency in characters and environment."
    )
    user_msg = f"Main prompt: '{main_prompt}'. Please provide {segment_count} segment prompts."
    content = _dashscope_text_chat(user_msg, system=system_msg, model="qwen-plus")
    if not content:
        return [main_prompt] * segment_count
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    if len(lines) < segment_count:
        return lines + [main_prompt] * (segment_count - len(lines))
    return lines[:segment_count]


def _write_output_bytes(filename: str, blob: bytes) -> Path:
    COMFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = COMFY_OUTPUT_DIR / filename
    path.write_bytes(blob)
    return path


def _upload_bytes_to_r2(s3_client, cfg: Dict, key: str, data: bytes, content_type: str) -> str:
    s3_client.put_object(Bucket=cfg["bucket"], Key=key, Body=data, ContentType=content_type)
    return f"{cfg['public_url']}/{key}"


def _local_model_path(kind: str, name: str) -> Path:
    folder = MODEL_KIND_TO_FOLDER[kind]
    return COMFY_ROOT / "models" / folder / name


def _local_model_meta_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".sync.json")


def _model_s3_key(kind: str, name: str) -> str:
    folder = MODEL_KIND_TO_FOLDER[kind]
    return f"{MODEL_S3_ROOT_PREFIX}/{folder}/{name}"


def _model_sync_client_and_cfg():
    if not MODEL_SYNC_ENABLED:
        return None, None
    access_key = _first_env("MODEL_S3_ACCESS_KEY_ID", "S3_ACCESS_KEY_ID", "R2_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
    secret_key = _first_env("MODEL_S3_SECRET_ACCESS_KEY", "S3_SECRET_ACCESS_KEY", "R2_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
    bucket = _first_env("MODEL_S3_BUCKET", "S3_BUCKET", "R2_BUCKET")
    endpoint = _first_env("MODEL_S3_ENDPOINT_URL", "S3_ENDPOINT_URL", "R2_ENDPOINT")
    region = _first_env("MODEL_S3_REGION", "S3_REGION", "R2_REGION", default="eu-ro-1")
    root_prefix = _first_env("MODEL_S3_ROOT_PREFIX", "S3_MODEL_ROOT_PREFIX", default="runpod-slim/ComfyUI/models").strip("/")
    account_id = _first_env("R2_ACCOUNT_ID")
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    if not (access_key and secret_key and bucket and endpoint):
        return None, None
    endpoint = endpoint.rstrip("/")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return s3, {
        "bucket": bucket,
        "root_prefix": root_prefix,
        "endpoint": endpoint,
    }


def _acquire_model_sync_lock():
    MODEL_SYNC_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = MODEL_SYNC_LOCK_PATH.open("w")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _s3_object_head(s3_client, bucket: str, key: str):
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        return None, exc


def _load_local_model_meta(model_path: Path) -> Dict:
    meta_path = _local_model_meta_path(model_path)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_local_model_meta(model_path: Path, meta: Dict) -> None:
    meta_path = _local_model_meta_path(model_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _download_s3_model(s3_client, bucket: str, key: str, dst: Path, head: Dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent), prefix=f".{dst.name}.") as tmp:
        tmp_path = Path(tmp.name)
        s3_client.download_fileobj(bucket, key, tmp)
    os.replace(tmp_path, dst)
    _save_local_model_meta(
        dst,
        {
            "bucket": bucket,
            "key": key,
            "etag": head.get("ETag"),
            "size": head.get("ContentLength"),
            "last_modified": head.get("LastModified").isoformat() if head.get("LastModified") else None,
        },
    )


def _ensure_model_available(s3_client, cfg: Dict, kind: str, name: str, warnings: List[str]) -> None:
    if kind not in MODEL_KIND_TO_FOLDER:
        raise RuntimeError(f"Unsupported model kind: {kind}")
    model_path = _local_model_path(kind, name)
    key = _model_s3_key(kind, name)
    head = None
    head_err = None
    try:
        head = s3_client.head_object(Bucket=cfg["bucket"], Key=key)
    except Exception as exc:
        head_err = exc

    if model_path.exists() and model_path.stat().st_size > 0:
        meta = _load_local_model_meta(model_path)
        if head and meta.get("etag") == head.get("ETag") and int(meta.get("size") or -1) == int(head.get("ContentLength") or -2):
            return
        if head and not meta:
            _save_local_model_meta(
                model_path,
                {
                    "bucket": cfg["bucket"],
                    "key": key,
                    "etag": head.get("ETag"),
                    "size": head.get("ContentLength"),
                    "last_modified": head.get("LastModified").isoformat() if head.get("LastModified") else None,
                },
            )
            return

        if not head:
            warnings.append(f"using cached {kind}:{name}")
            return

    if not head:
        raise RuntimeError(
            f"Model sync failed for {kind}:{name} at {key}: "
            f"{head_err if head_err else 'missing remote object'}"
        )

    warnings.append(f"synced {kind}:{name}")
    _download_s3_model(s3_client, cfg["bucket"], key, model_path, head)


def sync_requested_models(prompt: Dict, input_data: Dict) -> List[str]:
    if not MODEL_SYNC_ENABLED:
        return []
    s3_client, cfg = _model_sync_client_and_cfg()
    if not s3_client:
        return []

    specs: List[Tuple[str, str]] = []
    ckpt_name = ""
    if isinstance(prompt, dict):
        ckpt_node = prompt.get("1", {})
        if isinstance(ckpt_node, dict):
            ckpt_name = str(ckpt_node.get("inputs", {}).get("ckpt_name", "")).strip()
    if not ckpt_name:
        ckpt_name = str(input_data.get("ckpt_name", "")).strip()
    if ckpt_name:
        specs.append(("checkpoint", ckpt_name))
    if isinstance(prompt, dict):
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", "")).strip()
            inputs = node.get("inputs", {}) if isinstance(node.get("inputs", {}), dict) else {}
            if class_type == "LoraLoader":
                lora_name = str(inputs.get("lora_name", "")).strip()
                if lora_name:
                    specs.append(("lora", lora_name))
            elif class_type == "UpscaleModelLoader":
                upscale_name = str(inputs.get("model_name", "")).strip()
                if upscale_name:
                    specs.append(("upscale_model", upscale_name))
    if not specs and isinstance(input_data.get("loras"), list):
        for item in input_data["loras"]:
            if isinstance(item, dict) and str(item.get("name", "")).strip():
                specs.append(("lora", str(item["name"]).strip()))
    if not specs:
        upscale_name = str(input_data.get("upscale_model_name", "")).strip()
        if upscale_name and bool(input_data.get("enable_upscale", input_data.get("use_upscale", False))):
            specs.append(("upscale_model", upscale_name))

    if not specs:
        return []

    synced: List[str] = []
    lock_file = _acquire_model_sync_lock()
    try:
        for kind, name in sorted(set(specs)):
            _ensure_model_available(s3_client, cfg, kind, name, synced)
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
    return synced


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
    if str(input_data.get("startimg", "")).strip():
        return WAN_EXTEND_ANY_FRAME_MODE
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
    data.setdefault("qwen_swap_prompt", QWEN_DEFAULT_SWAP_PROMPT)
    data.setdefault("qwen_edit_prompt", QWEN_DEFAULT_EDIT_PROMPT)
    data.setdefault("qwen_model", QWEN_MODEL)
    data.setdefault("qwen_size", "")
    data.setdefault("qwen_extra_image", "")
    data.setdefault("startimg", "")
    data.setdefault("endimg", "")
    data.setdefault("frames", WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT)
    data.setdefault("enable_i2v", False)
    data.setdefault("i2v_prompt", data.get("prompt", I2V_DEFAULT_PROMPT))
    data.setdefault("i2v_model", I2V_MODEL)
    data.setdefault("i2v_resolution", "1080P")
    data.setdefault("i2v_duration", 5)
    data.setdefault("i2v_seed", 0)
    data.setdefault("i2v_negative_prompt", "")
    data.setdefault("i2v_audio_url", "")
    data.setdefault("i2v_prompt_extend", True)
    data.setdefault("i2v_watermark", False)
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
    if data["mode"] == WAN_EXTEND_ANY_FRAME_MODE:
        data["enable_upscale"] = False
    if "enable_pulid" not in data:
        data["enable_pulid"] = data["mode"] not in {"pose_only", "text_only", "qwen_swap_face", "qwen_edit_face"}
    if data["mode"] in {"qwen_swap_face", "qwen_edit_face"}:
        data["enable_pulid"] = False
    if data["mode"] == WAN_EXTEND_ANY_FRAME_MODE:
        data["enable_pulid"] = False
        data["enable_i2v"] = False
        data["prompt"] = str(data.get("prompt", "")).strip() or WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT
        data["frames"] = int(data.get("frames", WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT) or WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT)
        data.setdefault("width", WAN_DEFAULT_WIDTH)
        data.setdefault("height", WAN_DEFAULT_HEIGHT)
        data.setdefault("base_steps", 4)
        data.setdefault("steps", 4)
        data.setdefault("base_cfg", 2.0)
        data.setdefault("cfg", 1.0)
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
    elif mode == "qwen_swap_face":
        if "reference_image" not in input_data:
            raise RuntimeError("reference_image is required for qwen_swap_face")
    elif mode == WAN_EXTEND_ANY_FRAME_MODE:
        if not str(input_data.get("startimg", "")).strip():
            raise RuntimeError("startimg is required for wan2_2_i2v_extend_any_frame")
        if int(input_data.get("frames", 0) or 0) <= 0:
            raise RuntimeError("frames must be greater than 0 for wan2_2_i2v_extend_any_frame")
    elif mode == "qwen_edit_face":
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
    enable_pulid = bool(input_data.get("enable_pulid", mode not in {"pose_only", "text_only", "qwen_swap_face", "qwen_edit_face"}))
    if mode in {"qwen_swap_face", "qwen_edit_face"}:
        enable_pulid = False
    uses_external_pose = False
    if mode == "dual_pass_auto_pose":
        set_pose_branch(prompt, False)
        prune_nodes(prompt, ("9",))
    elif mode in {"pose_then_face_swap", "pose_only"}:
        set_pose_branch(prompt, True)
        uses_external_pose = True
        prune_nodes(prompt, ("22", "23", "27"))
    elif mode == "text_only" or mode in {"qwen_swap_face", "qwen_edit_face"}:
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

    if "22" in prompt:
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

    if "12" in prompt:
        if "cn_depth_strength" in input_data:
            prompt["12"]["inputs"]["strength"] = float(input_data["cn_depth_strength"])
        if "cn_depth_start_percent" in input_data:
            prompt["12"]["inputs"]["start_percent"] = float(input_data["cn_depth_start_percent"])
        if "cn_depth_end_percent" in input_data:
            prompt["12"]["inputs"]["end_percent"] = float(input_data["cn_depth_end_percent"])
    if "26" in prompt:
        if "cn_pose_strength" in input_data:
            prompt["26"]["inputs"]["strength"] = float(input_data["cn_pose_strength"])
        if "cn_pose_start_percent" in input_data:
            prompt["26"]["inputs"]["start_percent"] = float(input_data["cn_pose_start_percent"])
        if "cn_pose_end_percent" in input_data:
            prompt["26"]["inputs"]["end_percent"] = float(input_data["cn_pose_end_percent"])

    if "8" in prompt:
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
    if input_data.get("mode") in {"qwen_swap_face", "qwen_edit_face"}:
        enable_pulid = False
    prompt["14"]["inputs"]["model"] = ["8", 0] if enable_pulid else model_source
    if not enable_lora:
        prune_nodes(prompt, ("17",))
    if not enable_pulid:
        prune_nodes(prompt, ("4", "5", "6", "7", "8"))
    if input_data.get("mode") == "qwen_edit_face":
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


def wait_history(prompt_id: str, timeout_sec: int = 1200, event: Dict = None) -> Dict:
    deadline = time.time() + timeout_sec
    log_file = Path("/tmp/comfy.log")
    f = None
    try:
        import runpod
    except ImportError:
        runpod = None

    if event is not None and log_file.exists() and runpod:
        try:
            f = log_file.open("r", encoding="utf-8")
            f.seek(0, 2)
        except Exception:
            f = None

    try:
        while time.time() < deadline:
            r = requests.get(f"{COMFY_API_URL}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            data = r.json()
            if prompt_id in data and data[prompt_id].get("outputs"):
                return data[prompt_id]
            
            if f and runpod:
                try:
                    new_lines = f.read()
                    if new_lines:
                        try:
                            from runpod.serverless.modules.rp_progress import progress_update
                            progress_update(event, {"logs": new_lines})
                        except Exception:
                            # Fallback if the direct import fails
                            runpod.serverless.progress_update(event, {"logs": new_lines})
                except Exception:
                    pass
            time.sleep(1.5)
        raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")
    finally:
        if f:
            f.close()


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


def _extract_dashscope_image_url(resp_json: Dict) -> str:
    output = resp_json.get("output", {}) if isinstance(resp_json, dict) else {}
    choices = output.get("choices", []) if isinstance(output, dict) else []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                image_url = str(item.get("image", "")).strip()
                if image_url:
                    return image_url
    raise RuntimeError(f"DashScope response missing image URL: {json.dumps(resp_json, ensure_ascii=False)[:4000]}")


def _resolve_qwen_edit_prompt(template: str, prompt_text: str) -> str:
    source = template or QWEN_DEFAULT_EDIT_PROMPT
    source = source.replace("{{prompt}}", prompt_text)
    source = source.replace("{{生图提示词的主提示词变量}}", prompt_text)
    source = source.replace("{{生成提示词的主提示词变量}}", prompt_text)
    return source


def _call_dashscope_qwen_face_swap(base_media: str, face_media: str, prompt: str) -> Tuple[bytes, str]:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for qwen face swap")

    # 图1 为底图 (base_media), 图2 为脸部 (face_media)
    base_input, _ = _media_to_qwen_data_url(base_media)
    face_input, _ = _media_to_qwen_data_url(face_media)

    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": base_input},
                        {"image": face_input},
                        {"text": prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-DataInspection": '{"input":"disable", "output":"disable"}',
    }

    resp = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(f"Qwen face swap failed: {resp.text}")

    res_data = resp.json()
    choices = res_data.get("output", {}).get("choices", [])
    if not choices:
        raise RuntimeError(f"Qwen face swap returned no choices: {json.dumps(res_data)}")
    
    content = choices[0].get("message", {}).get("content", [])
    output_url = ""
    for item in content:
        if "image" in item:
            output_url = item["image"]
            break
            
    if not output_url:
        raise RuntimeError(f"Qwen face swap returned no image in content: {json.dumps(res_data)}")

    img_resp = requests.get(output_url, timeout=120)
    img_resp.raise_for_status()
    return img_resp.content, img_resp.headers.get("content-type", "image/png")


def _call_qwen_face_swap(
    reference_media: str,
    base_image_bytes: bytes,
    extra_media: str,
    swap_prompt: str,
    negative_prompt: str,
    model: str,
    size_override: str = "",
) -> bytes:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for qwen face edit/swap modes")
    reference_input = ""
    if reference_media.strip():
        reference_input, _ = _media_to_qwen_data_url(reference_media)
    base_input, base_size = _image_bytes_to_qwen_data_url(base_image_bytes)
    size = size_override.strip() or f"{base_size[0]}*{base_size[1]}"
    extra_input = ""
    if extra_media.strip():
        extra_input, _ = _media_to_qwen_data_url(extra_media)
    content = [{"image": base_input}]
    if reference_input:
        content.append({"image": reference_input})
    if extra_input:
        content.append({"image": extra_input})
    payload = {
        "model": model or QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [*content, {"text": swap_prompt or QWEN_DEFAULT_SWAP_PROMPT}],
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": negative_prompt or " ",
            "prompt_extend": False,
            "watermark": False,
            "size": size,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-DataInspection": QWEN_DATA_INSPECTION,
    }
    resp = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    image_url = _extract_dashscope_image_url(data)
    img_resp = requests.get(image_url, timeout=120)
    img_resp.raise_for_status()
    return img_resp.content


def _extract_dashscope_video_url(resp_json: Dict) -> str:
    output = resp_json.get("output", {}) if isinstance(resp_json, dict) else {}
    if isinstance(output, dict):
        video_url = str(output.get("video_url", "")).strip()
        if video_url:
            return video_url
    raise RuntimeError(f"DashScope response missing video URL: {json.dumps(resp_json, ensure_ascii=False)[:4000]}")


def _i2v_input_style_for_model(model_name: str) -> str:
    lowered = (model_name or "").strip().lower()
    if any(token in lowered for token in ("wan2.6", "wan2.5", "wan2.2", "wanx2.1")):
        return "img_url"
    return "media"


def _is_i2v_payload_shape_error(exc: Exception) -> bool:
    message = str(exc)
    return "Field required: input.media" in message or "Field required: input.img_url" in message


def _submit_and_wait_dashscope_i2v(payload: Dict, api_key: str, api_url: str) -> Tuple[bytes, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
        "X-DashScope-DataInspection": I2V_DATA_INSPECTION,
    }
    resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(f"DashScope i2v submit failed: status={resp.status_code}, body={resp.text[:4000]}")
    data = resp.json()
    task_id = str(data.get("output", {}).get("task_id", "")).strip()
    if not task_id:
        raise RuntimeError(f"DashScope i2v missing task_id: {json.dumps(data, ensure_ascii=False)[:4000]}")

    if "/services/aigc/video-generation/video-synthesis" in api_url:
        task_base = api_url.split("/services/aigc/video-generation/video-synthesis", 1)[0]
    else:
        task_base = api_url.rsplit("/video-synthesis", 1)[0]
    task_url = f"{task_base.rstrip('/')}/tasks/{task_id}"
    deadline = time.time() + 50 * 60
    while time.time() < deadline:
        time.sleep(15)
        poll = requests.get(task_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
        poll.raise_for_status()
        payload = poll.json()
        status = str(payload.get("output", {}).get("task_status", "")).strip().upper()
        if status == "SUCCEEDED":
            video_url = _extract_dashscope_video_url(payload)
            video_resp = requests.get(video_url, timeout=180)
            video_resp.raise_for_status()
            content_type = video_resp.headers.get("content-type", "").split(";")[0].strip()
            if not content_type:
                content_type = "video/mp4"
            return video_resp.content, content_type
        if status in {"FAILED", "CANCELED"}:
            raise RuntimeError(f"DashScope i2v task failed: {json.dumps(payload, ensure_ascii=False)[:4000]}")
    raise TimeoutError(f"DashScope i2v timed out for task_id {task_id}")


def _call_dashscope_i2v(
    base_image_bytes: bytes,
    prompt: str,
    negative_prompt: str,
    model: str,
    resolution: str,
    duration: int,
    seed: int,
    prompt_extend: bool,
    watermark: bool,
    audio_url: str,
) -> Tuple[bytes, str]:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for i2v")
    model_name = (model or I2V_MODEL).strip() or I2V_MODEL
    prompt_text = (prompt or I2V_DEFAULT_PROMPT).strip() or I2V_DEFAULT_PROMPT
    resolution_text = (resolution or "1080P").strip() or "1080P"
    base_input, _ = _image_bytes_to_qwen_data_url(base_image_bytes)
    api_url = I2V_API_URL.rstrip("/")
    if not api_url.endswith("/video-synthesis"):
        api_url = api_url + "/services/aigc/video-generation/video-synthesis"
    styles = [_i2v_input_style_for_model(model_name)]
    alternate = "img_url" if styles[0] == "media" else "media"
    styles.append(alternate)

    last_error: Optional[Exception] = None
    for input_style in styles:
        payload = {
            "model": model_name,
            "input": {
                "prompt": prompt_text,
            },
            "parameters": {
                "resolution": resolution_text,
                "duration": int(duration or 5),
                "prompt_extend": bool(prompt_extend),
                "watermark": bool(watermark),
            },
        }
        if input_style == "media":
            payload["input"]["media"] = [
                {
                    "type": "first_frame",
                    "url": base_input,
                }
            ]
        else:
            payload["input"]["img_url"] = base_input
        if negative_prompt.strip():
            payload["input"]["negative_prompt"] = negative_prompt.strip()
        if seed:
            payload["parameters"]["seed"] = int(seed)
        if audio_url.strip():
            payload["input"]["audio_url"] = audio_url.strip()

        try:
            return _submit_and_wait_dashscope_i2v(payload, api_key, api_url)
        except Exception as exc:
            last_error = exc
            if input_style != styles[-1] and _is_i2v_payload_shape_error(exc):
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("DashScope i2v failed without a specific error")


def _call_dashscope_i2v_extend_any_frame(
    first_media: str,
    prompt: str,
    negative_prompt: str,
    model: str,
    resolution: str,
    prompt_extend: bool,
    watermark: bool,
    audio_url: str,
    seed: int,
    last_media: str = "",
) -> Tuple[bytes, str]:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for wan2.2 i2v extend mode")

    model_name = (model or WAN_EXTEND_ANY_FRAME_MODEL).strip() or WAN_EXTEND_ANY_FRAME_MODEL
    prompt_text = (prompt or WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT).strip() or WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT
    resolution_text = (resolution or "720P").strip() or "720P"
    request_id = uuid.uuid4().hex
    first_input = _media_to_dashscope_accessible_url(first_media, request_id, "first_frame")
    last_input = ""
    if last_media.strip():
        last_input = _media_to_dashscope_accessible_url(last_media, request_id, "last_frame")

    api_url = I2V_API_URL.rstrip("/")
    if not api_url.endswith("/video-synthesis"):
        api_url = api_url + "/services/aigc/video-generation/video-synthesis"
    payload = {
        "model": model_name,
        "input": {
            "prompt": prompt_text,
            "first_frame_url": first_input,
        },
        "parameters": {
            "resolution": resolution_text,
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
            "seed": seed,
        },
    }
    if last_input:
        payload["input"]["last_frame_url"] = last_input
    if negative_prompt.strip():
        payload["input"]["negative_prompt"] = negative_prompt.strip()
    if audio_url.strip():
        payload["input"]["audio_url"] = audio_url.strip()

    return _submit_and_wait_dashscope_i2v(payload, api_key, api_url)


def _extract_video_last_frame_bytes(video_path: Path) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        frame_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(frame_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not frame_path.exists():
            raise RuntimeError(
                f"ffmpeg failed to extract last frame: {proc.stderr.decode('utf-8', errors='ignore')[:4000]}"
            )
        return frame_path.read_bytes()
    finally:
        try:
            frame_path.unlink(missing_ok=True)
        except Exception:
            pass


def _concat_video_segments(segment_paths: List[Path], output_path: Path) -> None:
    if not segment_paths:
        raise RuntimeError("No segment videos available to merge")
    if len(segment_paths) == 1:
        output_path.write_bytes(segment_paths[0].read_bytes())
        return

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8") as tmp:
        list_path = Path(tmp.name)
        for segment in segment_paths:
            safe = str(segment).replace("'", "'\\''")
            tmp.write(f"file '{safe}'\n")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and output_path.exists():
            return

        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
        proc = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"ffmpeg failed to merge segment videos: {proc.stderr.decode('utf-8', errors='ignore')[:4000]}"
            )
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass


def _make_video_from_frames(frame_paths: List[Path], output_path: Path, fps: int = WAN_VIDEO_FPS) -> None:
    if not frame_paths:
        raise RuntimeError("No frames available to encode video")

    with tempfile.TemporaryDirectory(prefix="wan_frames_") as tmpdir:
        tmp_dir = Path(tmpdir)
        for idx, frame_path in enumerate(frame_paths, start=1):
            target = tmp_dir / f"{idx:06d}.png"
            target.write_bytes(frame_path.read_bytes())
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(max(1, int(fps or WAN_VIDEO_FPS))),
            "-i",
            str(tmp_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"ffmpeg failed to encode video from frames: {proc.stderr.decode('utf-8', errors='ignore')[:4000]}"
            )


def _collect_node_images(history_obj: Dict, node_id: str) -> List[Dict]:
    outputs = history_obj.get("outputs", {}) if isinstance(history_obj, dict) else {}
    node_output = outputs.get(node_id, {}) if isinstance(outputs, dict) else {}
    images = node_output.get("images", []) if isinstance(node_output, dict) else []
    return [img for img in images if isinstance(img, dict)]


def _wan_backend_requested() -> str:
    backend = WAN_EXECUTION_BACKEND
    if backend in {"comfy", "dashscope", "auto"}:
        return backend
    return "auto"


def _wan_use_comfy_backend(data: Dict) -> bool:
    backend = _wan_backend_requested()
    if backend == "dashscope":
        return False
    if not WAN_WORKFLOW_API_PATH.exists():
        return False
    if backend == "comfy":
        return True
    if str(data.get("endimg", "")).strip():
        return False
    return True


def _apply_wan_lora_chain(prompt: Dict, loras: List[Dict], base_model: List = None) -> List:
    model_source = base_model if base_model else ["37", 0]
    if not loras:
        return model_source

    max_id = _max_node_id(prompt)
    for idx, lora in enumerate(loras):
        node_id = str(max_id + idx + 1)
        prompt[node_id] = {
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": float(lora.get("strength_model", lora.get("strength", 1.0) or 1.0)),
                "model": model_source,
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"WAN LoRA {idx + 1}"},
        }
        model_source = [node_id, 0]
    return model_source


def _wan_model_name(raw_value: object, fallback: str) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return fallback
    lower = value.lower()
    if "wan" in lower or "kf2v" in lower or "i2v" in lower or "svi" in lower:
        return value
    return fallback


def _apply_wan_workflow_defaults(prompt: Dict, data: Dict, current_start_image: str, segment_length: int, segment_idx: int) -> None:
    if "39" in prompt:
        prompt["39"]["inputs"]["vae_name"] = str(data.get("wan_vae_name", WAN_VAE_NAME)).strip() or WAN_VAE_NAME
    if "49" in prompt:
        prompt["49"]["inputs"]["clip_name"] = str(data.get("wan_clip_vision_name", WAN_CLIP_VISION_NAME)).strip() or WAN_CLIP_VISION_NAME
    if "38" in prompt:
        prompt["38"]["inputs"]["clip_name"] = str(data.get("wan_clip_name", WAN_CLIP_NAME)).strip() or WAN_CLIP_NAME
    if "37" in prompt:
        prompt["37"]["inputs"]["unet_name"] = _wan_model_name(data.get("wan_unet_high_name", WAN_UNET_HIGH_NAME), WAN_UNET_HIGH_NAME)
    if "100" in prompt:
        prompt["100"]["inputs"]["unet_name"] = _wan_model_name(data.get("wan_unet_low_name", WAN_UNET_LOW_NAME), WAN_UNET_LOW_NAME)
    if "52" in prompt:
        prompt["52"]["inputs"]["image"] = current_start_image
    if "6" in prompt:
        prompt["6"]["inputs"]["text"] = str(data.get("prompt", "")).strip() or WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT
    if "7" in prompt:
        prompt["7"]["inputs"]["text"] = str(data.get("negative_prompt", "")).strip()
    width = 0
    height = 0
    res_str = str(data.get("i2v_resolution", "")).strip().upper()
    if "*" in res_str:
        try:
            parts = res_str.split("*")
            width = int(parts[0])
            height = int(parts[1])
        except Exception:
            pass

    if not width or not height:
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))

    if not width or not height:
        try:
            input_path = COMFY_INPUT_DIR / current_start_image
            with Image.open(input_path) as img:
                img_w, img_h = img.size
            is_landscape = img_w > img_h
            if res_str == "1080P":
                width, height = (1920, 1088) if is_landscape else (1088, 1920)
            elif res_str == "720P":
                width, height = (1280, 720) if is_landscape else (720, 1280)
            else:
                width, height = (832, 480) if is_landscape else (480, 832)
        except Exception:
            width = int(WAN_DEFAULT_WIDTH)
            height = int(WAN_DEFAULT_HEIGHT)

    if "50" in prompt:
        prompt["50"]["inputs"]["width"] = width
        prompt["50"]["inputs"]["height"] = height
        prompt["50"]["inputs"]["length"] = int(segment_length)
        prompt["50"]["inputs"]["batch_size"] = 1

    base_steps = int(data.get("base_steps", 4))
    steps = int(data.get("steps", 4))
    total_steps = base_steps + steps
    base_cfg = float(data.get("base_cfg", 2.0))
    cfg = float(data.get("cfg", 1.0))

    if "102" in prompt:
        prompt["102"]["inputs"]["noise_seed"] = int(data.get("seed", 0) or 0)
        prompt["102"]["inputs"]["steps"] = total_steps
        prompt["102"]["inputs"]["end_at_step"] = base_steps
        prompt["102"]["inputs"]["cfg"] = base_cfg
    if "103" in prompt:
        prompt["103"]["inputs"]["noise_seed"] = int(data.get("seed", 0) or 0)
        prompt["103"]["inputs"]["steps"] = total_steps
        prompt["103"]["inputs"]["start_at_step"] = base_steps
        prompt["103"]["inputs"]["end_at_step"] = total_steps
        prompt["103"]["inputs"]["cfg"] = cfg
        
    if "47" in prompt:
        prompt["47"]["inputs"]["filename_prefix"] = f"wan_{data.get('request_id', 'wan')}_{segment_idx:02d}"



def _generate_wan_extend_any_frame_comfy(data: Dict, request_id: str, event: Dict = None) -> Dict:
    if not WAN_WORKFLOW_API_PATH.exists():
        raise RuntimeError(f"WAN workflow template not found: {WAN_WORKFLOW_API_PATH}")
    if str(data.get("endimg", "")).strip():
        raise RuntimeError("WAN Comfy workflow does not support endimg natively.")

    total_frames = int(data.get("frames", 81) or 81)
    segment_limit = 161  # 10 seconds at 16fps
    segment_count = max(1, (total_frames + segment_limit - 1) // segment_limit)

    prompt_text = str(data.get("prompt", "")).strip() or WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT
    negative_prompt = str(data.get("negative_prompt", "")).strip()
    auto_prompts = bool(data.get("auto_segment_prompts") or data.get("auto_generate_segment_prompts") or False)
    segment_prompts = (
        _generate_segment_prompts(prompt_text, segment_count)
        if auto_prompts
        else [prompt_text] * segment_count
    )

    loras = [
        x
        for x in (data.get("loras") or [])
        if isinstance(x, dict) and str(x.get("name", "")).strip()
    ]

    segment_video_paths = []
    current_start_media = str(data.get("startimg", "")).strip()

    print(f"[DEBUG-COMFY] Starting comfy processing. wan_face_swap={data.get('wan_face_swap')}", flush=True)

    if bool(data.get("wan_face_swap", False)):
        face_image = str(data.get("face_image", "")).strip()
        swap_prompt = str(data.get("wan_face_swap_prompt", QWEN_DEFAULT_SWAP_PROMPT)).strip()
        print(f"[DEBUG-COMFY] WAN Face Swap enabled. Face image present: {bool(face_image)}", flush=True)
        if face_image:
            try:
                # 图1 (startimg) 是底图, 图2 (face_image) 是脸部
                print(f"[DEBUG-COMFY] Calling Qwen face swap. Base image: {current_start_media[:100]}..., Face image: {face_image[:100]}...", flush=True)
                swapped_bytes, swapped_ct = _call_dashscope_qwen_face_swap(
                    current_start_media, face_image, swap_prompt
                )
                print(f"[DEBUG-COMFY] Qwen face swap completed. Swapped bytes size: {len(swapped_bytes)}", flush=True)
                current_start_media, _ = _image_bytes_to_qwen_data_url(swapped_bytes)
                
                # 同时也上传到 S3 方便前端展示验证
                s3, s3_cfg = get_r2_client_and_config()
                if s3:
                    swapped_key = f"intermediate/{request_id}/swapped_start.png"
                    swapped_url = upload_bytes_to_r2(s3, s3_cfg, swapped_key, swapped_bytes, swapped_ct or "image/png")
                    intermediate_urls.append(swapped_url)
                    print(f"[DEBUG-COMFY] Swapped image uploaded to: {swapped_url}", flush=True)
            except Exception as swap_exc:
                print(f"[WARN-COMFY] Face swap failed, falling back to original image: {swap_exc}", flush=True)

    for idx in range(segment_count):
        segment_idx = idx + 1
        current_prompt_text = segment_prompts[idx]
        segment_frames = min(segment_limit, total_frames - (idx * segment_limit))
        if segment_frames <= 0:
            break

        prompt = load_json(WAN_WORKFLOW_API_PATH)
        start_image_filename = resolve_media_to_comfy_filename(
            current_start_media, f"wan_start_{request_id}_{segment_idx}"
        )

        data_copy = dict(data)
        data_copy["seed"] = int(data.get("seed", 0) or 0) + idx
        _apply_wan_workflow_defaults(
            prompt, data_copy, start_image_filename, segment_frames, segment_idx
        )

        if "6" in prompt:
            prompt["6"]["inputs"]["text"] = current_prompt_text
        if "7" in prompt:
            prompt["7"]["inputs"]["text"] = negative_prompt

        # LORA LOGIC (Replicating okbox-comfy multi-lora high/low splitting)
        registry_path = Path("/runpod-volume/my_stable_models/lora_style_registry.json")
        registry = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if loras:
            next_node_id = 200
            prev_high_ref = ["37", 0]
            prev_low_ref = ["100", 0]
            for lora in loras:
                name = lora["name"]
                high_file = name
                low_file = name
                if name in registry:
                    high_file = registry[name].get("high", "none")
                    low_file = registry[name].get("low", "none")
                strength = float(lora.get("strength_model", lora.get("strength", 1.0) or 1.0))
                if high_file != "none":
                    node_id_h = str(next_node_id)
                    next_node_id += 1
                    prompt[node_id_h] = {
                        "inputs": {
                            "lora_name": high_file,
                            "strength_model": strength,
                            "model": prev_high_ref,
                        },
                        "class_type": "LoraLoaderModelOnly",
                    }
                    prev_high_ref = [node_id_h, 0]
                if low_file != "none":
                    node_id_l = str(next_node_id)
                    next_node_id += 1
                    prompt[node_id_l] = {
                        "inputs": {
                            "lora_name": low_file,
                            "strength_model": strength,
                            "model": prev_low_ref,
                        },
                        "class_type": "LoraLoaderModelOnly",
                    }
                    prev_low_ref = [node_id_l, 0]
            if "54" in prompt:
                prompt["54"]["inputs"]["model"] = prev_high_ref
            if "101" in prompt:
                prompt["101"]["inputs"]["model"] = prev_low_ref
        else:
            if "150" in prompt:
                prompt.pop("150", None)
            if "151" in prompt:
                prompt.pop("151", None)
            if "54" in prompt:
                prompt["54"]["inputs"]["model"] = ["37", 0]
            if "101" in prompt:
                prompt["101"]["inputs"]["model"] = ["100", 0]

        validate_required_node_types(prompt)
        prompt_id = queue_prompt(prompt)
        history_obj = wait_history(prompt_id, event=event)

        image_files = []
        outputs = history_obj.get("outputs", {})
        for nid, nout in outputs.items():
            if "images" in nout:
                for img_info in nout["images"]:
                    fname = img_info.get("filename", "")
                    subfolder = img_info.get("subfolder", "")
                    filepath = COMFY_OUTPUT_DIR / subfolder / fname
                    if filepath.exists():
                        image_files.append(filepath)

        if not image_files:
            raise RuntimeError(f"Segment {segment_idx} generated no frames.")

        image_files.sort()
        segment_filename = f"wan_{request_id}_seg_{segment_idx:02d}.mp4"
        segment_path = COMFY_OUTPUT_DIR / segment_filename

        if len(image_files) == 1:
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", str(image_files[0]),
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(segment_path),
            ]
        else:
            list_file = COMFY_OUTPUT_DIR / f"frames_{request_id}_{segment_idx}.txt"
            with list_file.open("w") as lf:
                for img_path in image_files:
                    lf.write(f"file '{img_path}'\n")
                    lf.write(f"duration {1.0/WAN_VIDEO_FPS}\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(segment_path),
            ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        segment_video_paths.append(segment_path)

        # Extract last frame for next segment start
        last_frame_bytes = image_files[-1].read_bytes()
        current_start_media, _ = _image_bytes_to_qwen_data_url(last_frame_bytes)

    if not segment_video_paths:
        raise RuntimeError("No segments were successfully generated.")

    merged_filename = f"wan_{request_id}_merged.mp4"
    merged_path = COMFY_OUTPUT_DIR / merged_filename
    _concat_video_segments(segment_video_paths, merged_path)

    s3, s3_cfg = get_r2_client_and_config()
    if not s3:
        return {
            "ok": True,
            "mode": WAN_EXTEND_ANY_FRAME_MODE,
            "request_id": request_id,
            "warning": "R2 env vars missing; returning local filenames only",
            "final_video_local_file": merged_filename,
            "final_video_local_files": [merged_filename],
        }

    merged_raw = merged_path.read_bytes()
    merged_url = _upload_bytes_to_r2(
        s3,
        s3_cfg,
        f"{s3_cfg['prefix']}/{request_id}/final_video.mp4",
        merged_raw,
        "video/mp4",
    )

    return {
        "ok": True,
        "mode": WAN_EXTEND_ANY_FRAME_MODE,
        "request_id": request_id,
        "final_video_url": merged_url,
        "final_video_urls": [merged_url],
        "meta": {
            "mode": WAN_EXTEND_ANY_FRAME_MODE,
            "frames": total_frames,
            "segment_count": segment_count,
            "prompt": prompt_text,
            "backend": "comfy",
            "workflow": str(WAN_WORKFLOW_API_PATH),
        },
    }


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


def _generate_wan_extend_any_frame(data: Dict, request_id: str) -> Dict:
    start_media = str(data.get("startimg", "")).strip()
    if not start_media:
        raise RuntimeError("startimg is required for wan2_2_i2v_extend_any_frame")
    end_media = str(data.get("endimg", "")).strip()
    prompt_text = str(data.get("prompt", "")).strip() or WAN_EXTEND_ANY_FRAME_DEFAULT_PROMPT
    negative_prompt = str(data.get("negative_prompt", "")).strip()
    resolution = str(data.get("i2v_resolution", "720P")).strip() or "720P"
    prompt_extend = bool(data.get("i2v_prompt_extend", True))
    watermark = bool(data.get("i2v_watermark", False))
    audio_url = str(data.get("i2v_audio_url", "")).strip()
    frames = int(data.get("frames", WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT) or WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT)
    segment_count = max(1, (frames + WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT - 1) // WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT)
    seed = int(data.get("seed", 0) or 0)

    auto_prompts = bool(data.get("auto_segment_prompts") or data.get("auto_generate_segment_prompts") or False)
    segment_prompts = (
        _generate_segment_prompts(prompt_text, segment_count)
        if auto_prompts
        else [prompt_text] * segment_count
    )

    segment_paths: List[Path] = []
    segment_records: List[Dict] = []
    intermediate_urls: List[str] = []
    current_start = start_media

    if bool(data.get("wan_face_swap", False)):
        face_image = str(data.get("face_image", "")).strip()
        swap_prompt = str(data.get("wan_face_swap_prompt", QWEN_DEFAULT_SWAP_PROMPT)).strip()
        print(f"[DEBUG] WAN Face Swap enabled. Face image present: {bool(face_image)}", flush=True)
        if face_image:
            try:
                # 图1 (startimg) 是底图, 图2 (face_image) 是脸部
                print(f"[DEBUG] Calling Qwen face swap. Base image: {current_start[:100]}..., Face image: {face_image[:100]}...", flush=True)
                swapped_bytes, swapped_ct = _call_dashscope_qwen_face_swap(
                    current_start, face_image, swap_prompt
                )
                print(f"[DEBUG] Qwen face swap completed. Swapped bytes size: {len(swapped_bytes)}", flush=True)
                current_start, _ = _image_bytes_to_qwen_data_url(swapped_bytes)
                
                # 同时也上传到 S3 方便前端展示验证
                s3, s3_cfg = get_r2_client_and_config()
                if s3:
                    swapped_key = f"intermediate/{request_id}/swapped_start.png"
                    swapped_url = upload_bytes_to_r2(s3, s3_cfg, swapped_key, swapped_bytes, swapped_ct or "image/png")
                    intermediate_urls.append(swapped_url)
                    print(f"[DEBUG] Swapped image uploaded to: {swapped_url}", flush=True)
            except Exception as swap_exc:
                print(f"[WARN] Face swap failed, falling back to original image: {swap_exc}", flush=True)

    for index in range(segment_count):
        current_prompt = segment_prompts[index]
        current_end = end_media if index == segment_count - 1 and end_media else ""
        video_raw, video_content_type = _call_dashscope_i2v_extend_any_frame(
            current_start,
            current_prompt,
            negative_prompt,
            WAN_EXTEND_ANY_FRAME_MODEL,
            resolution,
            prompt_extend,
            watermark,
            audio_url,
            seed + index,
            current_end,
        )
        segment_filename = f"wan_extend_{request_id}_{index + 1:02d}.mp4"
        segment_path = _write_output_bytes(segment_filename, video_raw)
        segment_paths.append(segment_path)
        segment_records.append(
            {
                "filename": segment_filename,
                "subfolder": "",
                "type": "output",
                "content_type": video_content_type or "video/mp4",
            }
        )
        last_frame_bytes = _extract_video_last_frame_bytes(segment_path)
        current_start, _ = _image_bytes_to_qwen_data_url(last_frame_bytes)

    merged_filename = f"wan_extend_{request_id}_merged.mp4"
    merged_path = COMFY_OUTPUT_DIR / merged_filename
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    _concat_video_segments(segment_paths, merged_path)

    s3, s3_cfg = get_r2_client_and_config()
    if not s3:
        return {
            "ok": True,
            "mode": WAN_EXTEND_ANY_FRAME_MODE,
            "request_id": request_id,
            "segment_count": segment_count,
            "warning": "R2 env vars missing; returning local filenames only",
            "final_video_local_file": merged_filename,
            "final_video_local_files": [merged_filename],
            "segment_video_local_files": [x["filename"] for x in segment_records],
        }

    segment_urls: List[str] = []
    for idx, segment_desc in enumerate(segment_records, start=1):
        raw = _read_output_image(segment_desc)
        key = f"{s3_cfg['prefix']}/{request_id}/segments/{idx:02d}_{segment_desc['filename']}"
        segment_urls.append(_upload_bytes_to_r2(s3, s3_cfg, key, raw, "video/mp4"))

    merged_raw = merged_path.read_bytes()
    merged_url = _upload_bytes_to_r2(
        s3,
        s3_cfg,
        f"{s3_cfg['prefix']}/{request_id}/final_video_01.mp4",
        merged_raw,
        "video/mp4",
    )

    return {
        "ok": True,
        "mode": WAN_EXTEND_ANY_FRAME_MODE,
        "request_id": request_id,
        "segment_count": segment_count,
        "final_video_url": merged_url,
        "final_video_urls": [merged_url],
        "segment_video_urls": segment_urls,
        "intermediate_urls": intermediate_urls,
        "meta": {
            "mode": WAN_EXTEND_ANY_FRAME_MODE,
            "frames": frames,
            "segment_count": segment_count,
            "segment_limit": WAN_EXTEND_ANY_FRAME_SEGMENT_LIMIT,
            "startimg": True,
            "endimg": bool(end_media),
            "prompt": prompt_text,
            "model": WAN_EXTEND_ANY_FRAME_MODEL,
        },
    }


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
    print(f"[DEBUG] Handler received mode: {data.get('mode')}", flush=True)
    request_id = data.get("request_id") or uuid.uuid4().hex
    data["request_id"] = request_id
    validate_input(data)

    print(f"[DEBUG] Mode comparison: {data['mode']} == {WAN_EXTEND_ANY_FRAME_MODE}?", flush=True)
    if data["mode"] == WAN_EXTEND_ANY_FRAME_MODE:
        print(f"[DEBUG] Extracted wan_face_swap: {data.get('wan_face_swap')} (type: {type(data.get('wan_face_swap'))})", flush=True)
        print(f"[DEBUG] Extracted face_image present: {bool(data.get('face_image'))}", flush=True)
        if _wan_use_comfy_backend(data):
            try:
                return _generate_wan_extend_any_frame_comfy(data, request_id, event=event)
            except Exception as exc:
                raise RuntimeError(f"WAN comfy backend failed: {exc}") from exc
        return _generate_wan_extend_any_frame(data, request_id)

    prompt = load_json(WORKFLOW_API_PATH)
    v3 = load_json(WORKFLOW_V3_PATH)
    apply_v3_defaults(prompt, v3)

    ref_value = data.get("reference_image")
    if ref_value and data["mode"] not in {"qwen_swap_face", "qwen_edit_face"}:
        ref_filename = resolve_media_to_comfy_filename(ref_value, "reference")
        prompt["7"]["inputs"]["image"] = ref_filename

    has_pose, enable_pulid = apply_mode(prompt, data)
    if has_pose:
        pose_filename = resolve_media_to_comfy_filename(data["pose_image"], "pose")
        prompt["9"]["inputs"]["image"] = pose_filename

    use_upscale, enable_pulid = apply_overrides(prompt, data)
    synced_models = sync_requested_models(prompt, data)
    keep_intermediate = bool(data.get("keep_intermediate", KEEP_INTERMEDIATE_DEFAULT))
    jpg_quality = int(data.get("jpg_quality", 85))

    validate_required_node_types(prompt)
    prompt_id = queue_prompt(prompt)
    history_obj = wait_history(prompt_id, event=event)
    final_images, intermediate_images = collect_output_images(history_obj)
    if not final_images:
        return {
            "ok": False,
            "error": "No final image produced by node 16",
            "prompt_id": prompt_id,
            "history_debug": _summarize_history(history_obj),
        }

    qwen_mode = data["mode"] in {"qwen_swap_face", "qwen_edit_face"}
    qwen_swap_prompt = str(data.get("qwen_swap_prompt", "")).strip()
    qwen_edit_prompt = str(data.get("qwen_edit_prompt", "")).strip()
    qwen_model = str(data.get("qwen_model", QWEN_MODEL)).strip() or QWEN_MODEL
    qwen_size = str(data.get("qwen_size", "")).strip()
    qwen_extra_image = str(data.get("qwen_extra_image", "")).strip()
    if qwen_mode:
        if keep_intermediate:
            intermediate_images = list(final_images) + list(intermediate_images)
        qwen_final_images: List[Dict] = []
        for idx, img_desc in enumerate(final_images, start=1):
            base_raw = _read_output_image(img_desc)
            qwen_raw = _call_qwen_face_swap(
                "" if data["mode"] == "qwen_edit_face" else ref_value,
                base_raw,
                "" if data["mode"] == "qwen_edit_face" else qwen_extra_image,
                _resolve_qwen_edit_prompt(qwen_edit_prompt, str(data.get("prompt", "")).strip()) if data["mode"] == "qwen_edit_face" else qwen_swap_prompt,
                str(data.get("negative_prompt", "")).strip(),
                qwen_model,
                qwen_size,
            )
            filename = f"qwen_swap_{request_id}_{idx:02d}.png"
            _write_output_bytes(filename, qwen_raw)
            qwen_final_images.append({"filename": filename, "subfolder": "", "type": "output"})
        final_images = qwen_final_images

    i2v_enabled = bool(data.get("enable_i2v"))
    i2v_final_videos: List[Dict] = []
    if i2v_enabled:
        i2v_prompt = str(data.get("i2v_prompt", "")).strip() or I2V_DEFAULT_PROMPT
        i2v_model = str(data.get("i2v_model", I2V_MODEL)).strip() or I2V_MODEL
        i2v_resolution = str(data.get("i2v_resolution", "1080P")).strip() or "1080P"
        i2v_duration = int(data.get("i2v_duration", 5) or 5)
        i2v_seed = int(data.get("i2v_seed", 0) or 0)
        i2v_negative_prompt = str(data.get("i2v_negative_prompt", "")).strip()
        i2v_audio_url = str(data.get("i2v_audio_url", "")).strip()
        i2v_prompt_extend = bool(data.get("i2v_prompt_extend", True))
        i2v_watermark = bool(data.get("i2v_watermark", False))
        for idx, img_desc in enumerate(final_images, start=1):
            base_raw = _read_output_image(img_desc)
            video_raw, video_content_type = _call_dashscope_i2v(
                base_raw,
                i2v_prompt,
                i2v_negative_prompt,
                i2v_model,
                i2v_resolution,
                i2v_duration,
                i2v_seed,
                i2v_prompt_extend,
                i2v_watermark,
                i2v_audio_url,
            )
            filename = f"i2v_{request_id}_{idx:02d}.mp4"
            _write_output_bytes(filename, video_raw)
            i2v_final_videos.append(
                {
                    "filename": filename,
                    "subfolder": "",
                    "type": "output",
                    "content_type": video_content_type,
                }
            )

    s3, s3_cfg = get_r2_client_and_config()
    if not s3:
        return {
            "ok": True,
            "prompt_id": prompt_id,
            "request_id": request_id,
            "warning": "R2 env vars missing; returning local filenames only",
            "final_local_file": final_images[0]["filename"],
            "final_local_files": [x["filename"] for x in final_images],
            "final_video_local_file": i2v_final_videos[0]["filename"] if i2v_final_videos else "",
            "final_video_local_files": [x["filename"] for x in i2v_final_videos],
            "intermediate_local_files": [x["filename"] for x in intermediate_images],
            "synced_models": synced_models,
        }

    final_urls: List[str] = []
    for idx, img_desc in enumerate(final_images, start=1):
        final_raw = _read_output_image(img_desc)
        final_ext = Path(img_desc["filename"]).suffix.lower() or ".png"
        final_content_type = "image/png" if final_ext == ".png" else "image/jpeg"
        if use_upscale and not qwen_mode:
            final_raw = _convert_to_jpg_bytes(final_raw, quality=jpg_quality)
            final_ext = ".jpg"
            final_content_type = "image/jpeg"
        key = f"{s3_cfg['prefix']}/{request_id}/final_{idx:02d}{final_ext}"
        final_urls.append(_upload_bytes_to_r2(s3, s3_cfg, key, final_raw, final_content_type))
    final_url = final_urls[0]

    final_video_urls: List[str] = []
    if i2v_enabled:
        for idx, video_desc in enumerate(i2v_final_videos, start=1):
            raw = _read_output_image(video_desc)
            ext = Path(video_desc["filename"]).suffix.lower() or ".mp4"
            content_type = "video/mp4"
            key = f"{s3_cfg['prefix']}/{request_id}/final_video_{idx:02d}{ext}"
            final_video_urls.append(_upload_bytes_to_r2(s3, s3_cfg, key, raw, content_type))

    intermediate_urls: List[str] = []
    if keep_intermediate:
        for idx, img_desc in enumerate(intermediate_images, start=1):
            raw = _read_output_image(img_desc)
            ext = Path(img_desc["filename"]).suffix.lower() or ".png"
            content_type = "image/png" if ext == ".png" else "image/jpeg"
            key = f"{s3_cfg['prefix']}/{request_id}/intermediate/{idx:02d}_{img_desc['filename']}"
            intermediate_urls.append(_upload_bytes_to_r2(s3, s3_cfg, key, raw, content_type))

    return {
        "ok": True,
        "prompt_id": prompt_id,
        "request_id": request_id,
        "storage": {"provider": "r2", "bucket": s3_cfg["bucket"]},
        "final_url": final_url,
        "final_urls": final_urls,
        "final_video_url": final_video_urls[0] if final_video_urls else "",
        "final_video_urls": final_video_urls,
        "intermediate_urls": intermediate_urls,
        "synced_models": synced_models,
        "meta": {
            "mode": data["mode"],
            "pose_mode": "external_pose"
            if has_pose
            else (
                "text_only"
                if data["mode"] == "text_only"
                else (
                    "qwen_swap_face"
                    if data["mode"] == "qwen_swap_face"
                    else ("qwen_edit_face" if data["mode"] == "qwen_edit_face" else "dual_pass_auto_pose")
                )
            ),
            "enable_pulid": enable_pulid,
            "enable_lora": bool(data.get("enable_lora")),
            "use_upscale": use_upscale,
            "qwen_model": qwen_model if qwen_mode else None,
            "qwen_edit_prompt": qwen_edit_prompt if data["mode"] == "qwen_edit_face" else None,
            "i2v": {
                "enabled": i2v_enabled,
                "model": i2v_model if i2v_enabled else None,
                "resolution": i2v_resolution if i2v_enabled else None,
                "duration": i2v_duration if i2v_enabled else None,
                "seed": i2v_seed if i2v_enabled else None,
                "prompt_extend": i2v_prompt_extend if i2v_enabled else None,
                "watermark": i2v_watermark if i2v_enabled else None,
                "audio_url": i2v_audio_url if i2v_enabled else None,
            },
            "final_format": "png" if qwen_mode else ("jpg" if use_upscale else final_ext.lstrip(".")),
            "jpg_quality": jpg_quality if use_upscale and not qwen_mode else None,
        },
    }
