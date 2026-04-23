#!/usr/bin/env python3
import configparser
import json
import os
import sys

import boto3


def load_cfg(path: str):
    parser = configparser.ConfigParser()
    parser.read(path)
    sec = parser["runpods3"]
    return {
        "access_key": sec["aws_access_key_id"],
        "secret_key": sec["aws_secret_access_key"],
        "bucket": sec["Bucket name"],
        "endpoint": sec["Endpoint URL"],
        "region": os.getenv("S3_REGION", "eu-ro-1"),
    }


def allowed(name: str, kind: str) -> bool:
    lower = name.lower()
    if name.startswith("put_"):
        return False
    if kind in {"checkpoint", "lora", "unet"}:
        return lower.endswith(".safetensors") or lower.endswith(".ckpt")
    if kind == "upscale_model":
        return lower.endswith(".pth") or lower.endswith(".pt")
    if kind == "vae":
        return lower.endswith(".safetensors") or lower.endswith(".pt") or lower.endswith(".ckpt")
    if kind in {"clip_vision", "text_encoder"}:
        return lower.endswith(".pt") or lower.endswith(".safetensors")
    return False


def list_prefix(s3, bucket: str, prefix: str, kind: str):
    out = []
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        name = key.rsplit("/", 1)[-1]
        if allowed(name, kind):
            out.append({"name": name, "path": key, "type": kind})
    return sorted(out, key=lambda item: item["name"].lower())


def main():
    creds = sys.argv[1]
    root_prefix = sys.argv[2].strip("/")
    cfg = load_cfg(creds)
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        region_name=cfg["region"],
    )
    result = {
        "checkpoints": list_prefix(s3, cfg["bucket"], f"{root_prefix}/checkpoints/", "checkpoint"),
        "unets": list_prefix(s3, cfg["bucket"], f"{root_prefix}/unet/", "unet"),
        "loras": list_prefix(s3, cfg["bucket"], f"{root_prefix}/loras/", "lora"),
        "upscale_models": list_prefix(s3, cfg["bucket"], f"{root_prefix}/upscale_models/", "upscale_model"),
        "vaes": list_prefix(s3, cfg["bucket"], f"{root_prefix}/vae/", "vae"),
        "clip_visions": list_prefix(s3, cfg["bucket"], f"{root_prefix}/clip_vision/", "clip_vision"),
        "text_encoders": list_prefix(s3, cfg["bucket"], f"{root_prefix}/text_encoders/", "text_encoder"),
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
