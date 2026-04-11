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
- Workflow pushes to GHCR by default.
- To enable Docker Hub push, set repo secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`
- Docker Hub image path will be:
  - `docker.io/<DOCKERHUB_USERNAME>/ponyv2:<tag>`

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

Notes:
- The YAML file is JSON-compatible YAML so the installer can parse it with Python stdlib only.
- Default `CIVITAI_TOKEN` in script comes from archived manifest and can be overridden via env.
- Set `INSTALL_CUSTOM_NODES=0` to skip custom node installation.
- Restart ComfyUI after install so newly added nodes are registered.

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
