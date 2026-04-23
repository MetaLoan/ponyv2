#!/usr/bin/env python3
"""Download a Civitai model/version and either upload it to S3 or save it locally.

Usage examples:
  python3 scripts/civitai_to_s3.py \
    "https://civitai.com/models/178167?modelVersionId=1071060"

  python3 scripts/civitai_to_s3.py \
    "https://civitai.com/models/178167?modelVersionId=1071060" \
    --kind lora

  KEY_FILE=/workspace/key.env python3 scripts/civitai_to_s3.py URL --kind lora

  python3 scripts/civitai_to_s3.py URL --kind checkpoint \
    --target-dir /workspace/runpod-slim/ComfyUI/models/checkpoints

Environment / key file:
  - civitai=... or CIVITAI_TOKEN=...
  - S3_ACCESS_KEY_ID / aws_access_key_id
  - S3_SECRET_ACCESS_KEY / aws_secret_access_key
  - S3_BUCKET / Bucket name
  - S3_ENDPOINT_URL / Endpoint URL
  - S3_REGION (optional)
  - S3_MODEL_ROOT_PREFIX (optional, default: runpod-slim/ComfyUI/models)
  - When using --target-dir, S3 settings are optional and the file is saved locally.
"""

from __future__ import annotations

import argparse
import hashlib
import configparser
import hmac
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests


DEFAULT_ROOT_PREFIX = "runpod-slim/ComfyUI/models"
DEFAULT_REGION = "eu-ro-1"
MODEL_KIND_TO_FOLDER = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "upscale_model": "upscale_models",
}


def load_key_env(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"key file not found: {path}")
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or k.startswith("#"):
            continue
        os.environ.setdefault(k, v)


def load_s3_credentials(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"S3 credential file not found: {path}")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "runpods3" not in parser:
        raise SystemExit("S3 credential file must contain a [runpods3] section")
    sec = parser["runpods3"]
    os.environ.setdefault("S3_ACCESS_KEY_ID", sec.get("aws_access_key_id", "").strip())
    os.environ.setdefault("S3_SECRET_ACCESS_KEY", sec.get("aws_secret_access_key", "").strip())
    os.environ.setdefault("S3_BUCKET", sec.get("Bucket name", "").strip())
    os.environ.setdefault("S3_ENDPOINT_URL", sec.get("Endpoint URL", "").strip())


def get_token() -> str:
    token = os.getenv("CIVITAI_TOKEN", "").strip()
    if token:
        return token
    token = os.getenv("civitai", "").strip()
    if token:
        return token
    key_file = os.getenv("KEY_FILE", "").strip()
    if key_file:
        load_key_env(key_file)
        token = os.getenv("CIVITAI_TOKEN", os.getenv("civitai", "")).strip()
        if token:
            return token
    raise SystemExit("missing Civitai token (set CIVITAI_TOKEN, civitai, or KEY_FILE)")


def get_s3_cfg():
    access_key = os.getenv("S3_ACCESS_KEY_ID", os.getenv("aws_access_key_id", "")).strip()
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY", os.getenv("aws_secret_access_key", "")).strip()
    bucket = os.getenv("S3_BUCKET", os.getenv("Bucket name", "")).strip()
    endpoint = os.getenv("S3_ENDPOINT_URL", os.getenv("Endpoint URL", "")).strip()
    region = os.getenv("S3_REGION", DEFAULT_REGION).strip()
    root_prefix = os.getenv("S3_MODEL_ROOT_PREFIX", DEFAULT_ROOT_PREFIX).strip("/")

    if not (access_key and secret_key and bucket and endpoint):
        raise SystemExit("missing S3 config (access key, secret key, bucket, endpoint)")

    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
        "endpoint": endpoint.rstrip("/"),
        "region": region or DEFAULT_REGION,
        "root_prefix": root_prefix or DEFAULT_ROOT_PREFIX,
    }


def _aws_sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_key(key: str) -> str:
    return "/".join(quote(part, safe="") for part in key.split("/"))


def _sigv4_headers(
    method: str,
    endpoint: str,
    region: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    key: str,
    payload_hash: str,
    content_type: str | None = None,
    content_length: int | None = None,
) -> tuple[str, dict]:
    parsed = urlparse(endpoint)
    host = parsed.netloc
    path = f"/{bucket}/{_canonical_key(key)}"
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    canonical_headers = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if content_type:
        canonical_headers["content-type"] = content_type
    if content_length is not None:
        canonical_headers["content-length"] = str(content_length)

    signed_header_names = sorted(canonical_headers.keys())
    canonical_headers_str = "".join(f"{name}:{canonical_headers[name].strip()}\n" for name in signed_header_names)
    signed_headers = ";".join(signed_header_names)
    canonical_request = "\n".join(
        [
            method,
            path,
            "",
            canonical_headers_str,
            signed_headers,
            payload_hash,
        ]
    )
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    k_date = _aws_sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _aws_sign(k_date, region)
    k_service = _aws_sign(k_region, "s3")
    k_signing = _aws_sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    headers = {
        "Host": host,
        "X-Amz-Date": amz_date,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": authorization,
    }
    if content_type:
        headers["Content-Type"] = content_type
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return f"{parsed.scheme}://{host}{path}", headers


def extract_model_version_id(source: str) -> str:
    parsed = urlparse(source)
    qs = parse_qs(parsed.query)
    if "modelVersionId" in qs and qs["modelVersionId"]:
        return qs["modelVersionId"][0].strip()

    m = re.search(r"/api/download/models/(\d+)", source)
    if m:
        return m.group(1)

    m = re.search(r"/models/(\d+)", source)
    if m:
        return m.group(1)

    if source.isdigit():
        return source

    raise SystemExit(f"could not extract modelVersionId from: {source}")


def normalize_filename_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "model"


def get_civitai_version_info(model_version_id: str, token: str) -> dict:
    api_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
    resp = requests.get(api_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise SystemExit(f"unexpected Civitai version response for {model_version_id}")
    return data


def build_default_target_name(model_version_id: str, token: str, suffix: str = "") -> str:
    info = get_civitai_version_info(model_version_id, token)
    model_name = ""
    version_name = ""
    if isinstance(info.get("model"), dict):
        model_name = str(info["model"].get("name", "")).strip()
    version_name = str(info.get("name", "")).strip()
    parts = [normalize_filename_part(model_name) if model_name else "", normalize_filename_part(version_name) if version_name else ""]
    parts = [p for p in parts if p and p.lower() != "model"]
    if not parts:
        parts = [f"civitai_{model_version_id}"]
    stem = "_".join(parts)
    if suffix:
        stem = f"{stem}{suffix}"
    return stem


def fetch_download_name(model_version_id: str, token: str) -> str:
    api_url = f"https://civitai.com/api/download/models/{model_version_id}"
    resp = requests.get(api_url, headers={"Authorization": f"Bearer {token}"}, allow_redirects=False, timeout=60)
    if resp.status_code not in {302, 303, 307, 308, 200}:
        raise SystemExit(f"unexpected Civitai status {resp.status_code} for {api_url}")
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r'filename="?([^\";]+)"?', cd, re.I)
    if m:
        return m.group(1).strip()
    return f"civitai_{model_version_id}.safetensors"


def download_file(model_version_id: str, token: str, dst_dir: Path) -> Path:
    api_url = f"https://civitai.com/api/download/models/{model_version_id}"
    filename = fetch_download_name(model_version_id, token)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename
    tmp_path: Path | None = None
    with requests.get(
        api_url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=120,
    ) as resp:
        try:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dst_dir), prefix=f".{dst.name}.") as tmp:
                tmp_path = Path(tmp.name)
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
            os.replace(tmp_path, dst)
            return dst
        except Exception:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise


def rename_downloaded_file(local_file: Path, model_version_id: str, token: str) -> Path:
    target_stem = build_default_target_name(model_version_id, token)
    suffix = local_file.suffix.lower() or ".safetensors"
    target_name = f"{target_stem}{suffix}"
    if local_file.name == target_name:
        return local_file
    target = local_file.with_name(target_name)
    if target.exists() and target.resolve() != local_file.resolve():
        target.unlink()
    local_file.rename(target)
    return target


def upload_to_s3(local_file: Path, cfg: dict, kind: str) -> tuple[str, dict]:
    if kind not in MODEL_KIND_TO_FOLDER:
        raise SystemExit(f"unsupported kind: {kind}")
    folder = MODEL_KIND_TO_FOLDER[kind]
    key = f"{cfg['root_prefix']}/{folder}/{local_file.name}"
    size = local_file.stat().st_size
    url, headers = _sigv4_headers(
        method="PUT",
        endpoint=cfg["endpoint"],
        region=cfg["region"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        bucket=cfg["bucket"],
        key=key,
        payload_hash="UNSIGNED-PAYLOAD",
        content_type="application/octet-stream",
        content_length=size,
    )
    with local_file.open("rb") as fh:
        resp = requests.put(url, data=fh, headers=headers, timeout=3600)
    if resp.status_code >= 400:
        raise SystemExit(f"S3 upload failed: status={resp.status_code} body={resp.text[:1000]}")
    head_url, head_headers = _sigv4_headers(
        method="HEAD",
        endpoint=cfg["endpoint"],
        region=cfg["region"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        bucket=cfg["bucket"],
        key=key,
        payload_hash="UNSIGNED-PAYLOAD",
    )
    head = requests.head(head_url, headers=head_headers, timeout=60)
    if head.status_code >= 400:
        raise SystemExit(f"S3 head failed: status={head.status_code} body={head.text[:1000]}")
    return key, {"ETag": head.headers.get("ETag"), "ContentLength": int(head.headers.get("Content-Length", "0"))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a Civitai model/version and upload it to S3 or save locally")
    p.add_argument("source", help="Civitai model URL, download URL, or modelVersionId")
    p.add_argument("--kind", choices=sorted(MODEL_KIND_TO_FOLDER.keys()), default="lora", help="S3 folder type")
    p.add_argument("--download-dir", default="/tmp", help="temporary download directory")
    p.add_argument("--key-file", default=os.getenv("KEY_FILE", ""), help="optional key.env style file")
    p.add_argument("--s3-key-file", default=os.getenv("S3_KEY_FILE", ""), help="optional s3-credentials.txt style file")
    p.add_argument("--name", default="", help="override the uploaded filename stem")
    p.add_argument("--target-dir", default="", help="save the downloaded model to this local directory instead of uploading to S3")
    p.add_argument("--keep-local", action="store_true", help="keep the downloaded file after upload")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.key_file:
        load_key_env(args.key_file)
    if args.s3_key_file and not args.target_dir.strip():
        load_s3_credentials(args.s3_key_file)

    token = get_token()
    model_version_id = extract_model_version_id(args.source)
    download_dir = Path(args.download_dir)

    local_file: Path | None = None
    try:
        local_file = download_file(model_version_id, token, download_dir)
        if args.name.strip():
            desired_suffix = local_file.suffix.lower() or ".safetensors"
            desired = local_file.with_name(f"{normalize_filename_part(args.name)}{desired_suffix}")
            if desired.exists() and desired.resolve() != local_file.resolve():
                desired.unlink()
            local_file.rename(desired)
            local_file = desired
        else:
            local_file = rename_downloaded_file(local_file, model_version_id, token)
        if args.target_dir.strip():
            target_dir = Path(args.target_dir.strip())
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / local_file.name
            if target.exists() and target.resolve() != local_file.resolve():
                target.unlink()
            local_file.replace(target)
            print(f"[saved] {target}")
        else:
            s3_cfg = get_s3_cfg()
            key, head = upload_to_s3(local_file, s3_cfg, args.kind)
            print(f"[downloaded] {local_file}")
            print(f"[uploaded] bucket={s3_cfg['bucket']} key={key}")
            print(f"[etag] {head.get('ETag')}")
            print(f"[size] {head.get('ContentLength')}")
    finally:
        if local_file and local_file.exists() and not args.keep_local and not args.target_dir.strip():
            local_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
