# ponyv2

RunPod / ComfyUI one-shot model installer for V16 archived stack.
Installer v2 includes:
- V16 model downloads from YAML manifest
- Auto install/update of `comfyui_controlnet_aux` and `PuLID_ComfyUI`
- Auto pip install of node requirements and runtime deps (`onnxruntime-gpu`, `insightface`)

## Verified baseline

The current serverless build was stabilized around these changes:
- Python `3.10`
- CUDA base image `12.4.1`
- PyTorch wheels from `cu124`
- `numpy==1.26.4`
- `scipy==1.11.4`
- `onnx==1.18.0`
- `onnxruntime-gpu==1.19.2`
- `insightface==0.7.3`

Recent fixes that matter operationally:
- `b4dc551` reverted to the stable Python 3.10 / CUDA 12.4 path
- `d5e2ca4` aligned custom-node dependency installation order with the reference pony build
- `3b4f24b` fixed mixed NumPy installs by uninstalling the runtime stack before reinstall
- `6f16ea4` aligned ONNX Runtime GPU with CUDA 12 so InsightFace can use `CUDAExecutionProvider`
- `464ad4a` made R2 config accept both `R2_ACCOUNT_ID` and `R2_ENDPOINT`
- `qwen_swap_face` adds a DashScope Qwen image-edit face-swap post-process on top of the existing Comfy workflow

## One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/MetaLoan/ponyv2/main/scripts/install_v16_models.sh | bash -s -- /workspace/ComfyUI
```

## Optional: override Civitai token

```bash
export CIVITAI_TOKEN="your_token"
curl -fsSL https://raw.githubusercontent.com/MetaLoan/ponyv2/main/scripts/install_v16_models.sh | bash -s -- /workspace/ComfyUI
```

## Optional: load token from key.env

```bash
export KEY_ENV_FILE="/workspace/key.env"
curl -fsSL https://raw.githubusercontent.com/MetaLoan/ponyv2/main/scripts/install_v16_models.sh | bash -s -- /workspace/ComfyUI
```

## Model link manifest (YAML)

- `config/v16_models.yaml`

## PuLID diagnose (remote)

```bash
curl -fsSL https://raw.githubusercontent.com/MetaLoan/ponyv2/main/scripts/diagnose_pulid.sh | bash -s -- /workspace/runpod-slim/ComfyUI
```

## Serverless build (GitHub Actions)

- Workflow file: `.github/workflows/build-image.yml`
- Default image tag: `rpd-svls-sdxl-v0.1`
- Image registry: `ghcr.io/<owner>/ponyv2:<tag>`

Manual trigger:
1. Open `Actions` -> `Build And Push Serverless Image`
2. Click `Run workflow`
3. Keep default tag `rpd-svls-sdxl-v0.1` or override

Notes:
- Dockerfile supports full packaging switch:
  - `--build-arg BUILD_FULL_PACKAGE=1`
  - optional `--build-arg CIVITAI_TOKEN=<token>`
- Full packaging downloads models during image build and will be very large.
- Workflow pushes to GHCR only.

## R2 environment variables

The handler accepts either of these configuration styles:

Style A:
- `R2_ACCESS_KEY`
- `R2_SECRET_KEY`
- `R2_ACCOUNT_ID`
- `R2_BUCKET`
- `R2_PUBLIC_URL`

Style B:
- `R2_ACCESS_KEY`
- `R2_SECRET_KEY`
- `R2_ENDPOINT`
- `R2_BUCKET`
- `R2_PUBLIC_URL`

When both are present, `R2_ENDPOINT` is used first. `R2_PUBLIC_URL` must be the public CDN/base URL used in returned file URLs.

## Qwen edit/swap env vars

The RunPod handler supports post-process Qwen image edit modes for both face swap and prompt-driven face edit.

Required for `mode=qwen_swap_face` and `mode=qwen_edit_face`:

- `DASHSCOPE_API_KEY`

Optional overrides:

- `DASHSCOPE_API_URL`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_MODEL`
- `DASHSCOPE_DATA_INSPECTION_HEADER`

Default behavior:
- API endpoint defaults to the Singapore DashScope image-edit endpoint
- Model defaults to `qwen-image-edit-max`
- Data inspection header defaults to `{"input":"disable", "output":"disable"}`
- The worker sends the base-generated image as image 1, the user reference face as image 2, and an optional third image as image 3

## I2V post-process env vars

The RunPod worker can optionally turn every final image into a short video using DashScope image-to-video.

Required:

- `DASHSCOPE_API_KEY`

Optional overrides:

- `DASHSCOPE_I2V_API_URL`
- `DASHSCOPE_I2V_MODEL`
- `DASHSCOPE_DATA_INSPECTION_HEADER`

Default behavior:

- API endpoint defaults to the Singapore DashScope video endpoint
- Model defaults to `wan2.7-i2v`
- Data inspection header defaults to `{"input":"disable","output":"disable"}`
- If `enable_i2v` is turned on, the worker sends each final image as the first frame of a separate video task
- This is the same key used by `qwen_swap_face`; the worker also falls back to `QWEN_API_KEY` if you prefer that naming

Notes:
- The YAML file is JSON-compatible YAML so the installer can parse it with Python stdlib only.
- Default `CIVITAI_TOKEN` in script comes from archived manifest and can be overridden via env.
- Set `INSTALL_CUSTOM_NODES=0` to skip custom node installation.
- Restart ComfyUI after install so newly added nodes are registered.

## On-demand model sync in workers

The RunPod handler can lazily pull missing models from S3 before a job starts.

Supported env vars:
- `MODEL_SYNC_ON_DEMAND=1` to enable the feature
- `MODEL_S3_ACCESS_KEY_ID`
- `MODEL_S3_SECRET_ACCESS_KEY`
- `MODEL_S3_BUCKET`
- `MODEL_S3_ENDPOINT_URL`
- `MODEL_S3_REGION`
- `MODEL_S3_ROOT_PREFIX`

Behavior:
- The handler inspects the final rendered prompt and syncs the checkpoint, LoRA, and upscale model names that are actually referenced.
- A local cache metadata file is stored next to each downloaded model as `<model>.sync.json`.
- If a worker already has the file locally, it will reuse the cached file.
- If the remote object has changed, the worker refreshes it using the S3 object `ETag` and size as the cache key.
- If the remote sync cannot be checked but the local file exists, the worker keeps using the local copy instead of failing the job.

This gives you incremental model rollout without rebuilding the image, while still failing fast if a requested model is missing everywhere.

## One-off Civitai to S3 upload

Use the helper script when you want to paste a Civitai model page URL and upload the resolved file directly into the shared model bucket:

```bash
KEY_FILE=/workspace/key.env \
python3 scripts/civitai_to_s3.py "https://civitai.com/models/178167?modelVersionId=1071060" --kind lora
```

If you only have the token in your shell, you can also export it directly:

```bash
export CIVITAI_TOKEN="your_token"
python3 scripts/civitai_to_s3.py "https://civitai.com/models/178167?modelVersionId=1071060" --kind lora
```

If you want to use the local credential files directly:

```bash
KEY_FILE=/Users/leo/Desktop/sdxl2img/key.env \
S3_KEY_FILE=/Users/leo/Desktop/sdxl2img/s3-credentials.txt \
python3 scripts/civitai_to_s3.py "https://civitai.com/models/178167?modelVersionId=1071060" --kind lora
```

Interactive shell wrapper:

```bash
bash scripts/civitai_to_s3.sh
```

Supported kinds:
- `lora`
- `checkpoint`
- `upscale_model`

The script extracts `modelVersionId`, downloads the model with your Civitai token, and uploads it to:
- `runpod-slim/ComfyUI/models/loras/`
- `runpod-slim/ComfyUI/models/checkpoints/`
- `runpod-slim/ComfyUI/models/upscale_models/`

By default it renames the downloaded file to a normalized `model_name_version_name.ext` pattern. You can override that with `--name`.

## Fly.io web tester

This repo now includes a Fly-specific deployment for the React + Go tester:

- `Dockerfile.fly`
- `fly.toml`
- `DEPLOY_FLY.md`

Build:

```bash
fly deploy -c fly.toml
```

Required runtime env for generation:
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`

Required runtime env for the S3 model catalog:
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_BUCKET`
- `S3_ENDPOINT_URL`

The Fly app serves the React UI and the Go API from one container.
