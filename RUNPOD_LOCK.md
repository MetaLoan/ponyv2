# RunPod Lock

Captured from the working RunPod environment at `/workspace/runpod-slim/ComfyUI`.

This file is now split into:
- the original manual RunPod capture, kept for reference
- the serverless image baseline that has been validated in this repo

## Runtime

- Python: `3.12.3`
- ComfyUI repo: `e56b199` on `master`

## Key Python Packages

- `numpy==2.4.4`
- `scipy==1.17.1`
- `onnx==1.21.0`
- `onnxruntime==1.24.4`
- `onnxruntime-gpu==1.24.4`
- `insightface==0.7.3`
- `torch==2.10.0+cu128`
- `torchvision==0.25.0+cu128`
- `torchaudio==2.10.0+cu128`
- `facexlib==0.3.0`
- `ftfy==6.3.1`
- `timm==1.0.26`
- `aiohttp==3.13.3`
- `einops==0.8.2`
- `filelock==3.20.0`
- `multidict==6.7.1`
- `networkx==3.6.1`
- `opencv-python==4.13.0.92`
- `opencv-python-headless==4.13.0.92`
- `packaging==26.0`
- `psutil==7.2.2`
- `pydantic==2.12.5`
- `requests==2.32.5`
- `safetensors==0.7.0`
- `six==1.17.0`
- `sympy==1.14.0`
- `transformers==5.3.0`
- `triton==3.6.0`
- `urllib3==2.6.3`
- `websocket-client==1.9.0`
- `yarl==1.23.0`

## Custom Nodes

- `comfyui_controlnet_aux` - `main` - `95a13e2`
- `PuLID_ComfyUI` - `main` - `93e0c4c`
- `ComfyUI-VideoHelperSuite` - `main`
- `ComfyUI-KJNodes` - `master` - `d2ea1a7`
- `ComfyUI-Frame-Interpolation` - `main`
- `ComfyUI-Easy-Use` - `main`
- `cg-use-everywhere` - `main`
- `rgthree-comfy` - `main`
- `ComfyUI-Manager` - `master` - `863ffdc`
- `ComfyUI-RunpodDirect` - `master` - `e81cf6b`
- `Civicomfy` - `master` - `f8b82b3`

## Notes

- The captured environment is not a minimal mirror of the Docker image; it also contains many auxiliary notebook and ML packages.
- The Docker image should keep the core runtime aligned to the packages above, but exact full `pip freeze` parity is not required.

## Serverless Image Baseline

Validated working path in this repo:

- Base image: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- Python: `3.10`
- PyTorch install source: `https://download.pytorch.org/whl/cu124`
- Core runtime pins:
  - `numpy==1.26.4`
  - `scipy==1.11.4`
  - `onnx==1.18.0`
  - `onnxruntime-gpu==1.19.2`
  - `insightface==0.7.3`
  - `facexlib==0.3.0`
  - `ftfy==6.3.1`
  - `timm==1.0.26`

## Verified Runtime Findings

- The earlier `--ignore-installed` flow could leave mixed NumPy / SciPy / ONNX files behind and produce:
  - `module 'numpy.core.multiarray' has no attribute 'number'`
- Cleaning the runtime stack first and then reinstalling fixed that issue.
- `onnxruntime-gpu==1.18.0` fell back to CPU on the CUDA 12.4 image because it looked for `libcublasLt.so.11`.
- `onnxruntime-gpu==1.19.2` resolved that mismatch and produced:
  - `Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']`
- A full PuLID SDXL request was verified with final log line:
  - `Prompt executed in 193.30 seconds`

## R2 Compatibility

`app/handler.py` accepts either:
- `R2_ACCOUNT_ID`
- or `R2_ENDPOINT`

along with:
- `R2_ACCESS_KEY`
- `R2_SECRET_KEY`
- `R2_BUCKET`
- `R2_PUBLIC_URL`

The `R2_ENDPOINT` path was added so existing RunPod environment variable layouts do not need to be renamed.
