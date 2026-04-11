# Fly.io Deploy

This repository now includes a Fly-specific web deployment.

Files:
- `Dockerfile.fly`
- `fly.toml`
- `.dockerignore`

What it deploys:
- React frontend built from `frontend/`
- Go web service from `cmd/v16web`

Runtime notes:
- The app serves the frontend and API from one process.
- `generate` uses `COMFY_API_URL` or RunPod credentials if configured.
- Model catalog uses S3 credentials if configured.

Required environment variables for generation:
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`

Required environment variables for S3-backed model catalog:
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_BUCKET`
- `S3_ENDPOINT_URL`

Optional environment variables:
- `S3_REGION`
- `S3_MODEL_ROOT_PREFIX`
- `COMFY_API_URL`
- `V16WEB_ADDR`

Typical deploy flow:

```bash
cd /Users/leo/Desktop/sdxl2img/ponyv2
fly deploy -c fly.toml
```

If you want a different Fly app name, update `app` in `fly.toml` or pass `-a <app-name>`.
