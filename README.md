# ponyv2

RunPod / ComfyUI one-shot model installer for V16 archived stack.
Installer v2 includes:
- V16 model downloads from YAML manifest
- Auto install/update of `comfyui_controlnet_aux` and `PuLID_ComfyUI`
- Auto pip install of node requirements and runtime deps (`onnxruntime-gpu`, `insightface`)

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

Notes:
- The YAML file is JSON-compatible YAML so the installer can parse it with Python stdlib only.
- Default `CIVITAI_TOKEN` in script comes from archived manifest and can be overridden via env.
- Set `INSTALL_CUSTOM_NODES=0` to skip custom node installation.
- Restart ComfyUI after install so newly added nodes are registered.
