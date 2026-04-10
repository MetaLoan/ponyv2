#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]-}"
if [[ -n "$SCRIPT_PATH" && -f "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

REPO_RAW_BASE="${REPO_RAW_BASE:-https://raw.githubusercontent.com/MetaLoan/ponyv2/main}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/../config/v16_models.yaml}"
COMFY_ROOT="${1:-${COMFY_ROOT:-/workspace/ComfyUI}}"
CUSTOM_NODES_ROOT="${CUSTOM_NODES_ROOT:-$COMFY_ROOT/custom_nodes}"
KEY_ENV_FILE="${KEY_ENV_FILE:-$SCRIPT_DIR/../key.env}"
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"
DEFAULT_CIVITAI_TOKEN="fd0f3beec0b56c19715e0161cca7505c"
INSTALL_CUSTOM_NODES="${INSTALL_CUSTOM_NODES:-1}"

load_key_env_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0

  while IFS='=' read -r key value; do
    [[ -n "${key:-}" ]] || continue
    [[ "$key" =~ ^[A-Za-z0-9_-]+$ ]] || continue
    case "$key" in
      civitai|CIVITAI_TOKEN)
        if [[ -z "$CIVITAI_TOKEN" ]]; then
          CIVITAI_TOKEN="$value"
        fi
        ;;
    esac
  done < "$f"
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERR] python3 is required" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  tmp_cfg="/tmp/v16_models.yaml"
  echo "[INFO] Config not found locally, fetching from repo raw..."
  curl -fL --retry 3 --retry-delay 2 -o "$tmp_cfg" "$REPO_RAW_BASE/config/v16_models.yaml"
  CONFIG_FILE="$tmp_cfg"
fi

load_key_env_file "$KEY_ENV_FILE"
if [[ -z "$CIVITAI_TOKEN" ]]; then
  CIVITAI_TOKEN="$DEFAULT_CIVITAI_TOKEN"
fi

download_file() {
  local url="$1"
  local dst="$2"

  mkdir -p "$(dirname "$dst")"

  if [[ -s "$dst" ]]; then
    echo "[SKIP] $dst already exists"
    return 0
  fi

  echo "[GET ] $url"
  echo "[TO  ] $dst"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -k 1M --allow-overwrite=true -o "$(basename "$dst")" -d "$(dirname "$dst")" "$url"
  else
    curl -fL --retry 3 --retry-delay 2 -o "$dst" "$url"
  fi
}

install_requirements_if_exists() {
  local req_file="$1"
  [[ -f "$req_file" ]] || return 0
  echo "[PIP ] $req_file"
  python3 -m pip install -r "$req_file"
}

ensure_git_repo() {
  local repo_url="$1"
  local repo_dir="$2"
  local branch="${3:-}"

  if [[ -d "$repo_dir" && ! -d "$repo_dir/.git" ]]; then
    local backup_dir="${repo_dir}.preclone.$(date +%s)"
    echo "[WARN] $repo_dir exists but is not a git repo, moving to $backup_dir"
    mv "$repo_dir" "$backup_dir"
  fi

  if [[ ! -d "$repo_dir/.git" ]]; then
    echo "[GIT ] clone $repo_url -> $repo_dir"
    if [[ -n "$branch" ]]; then
      git clone --depth 1 -b "$branch" "$repo_url" "$repo_dir"
    else
      git clone --depth 1 "$repo_url" "$repo_dir"
    fi
    return 0
  fi

  echo "[GIT ] update $repo_dir"
  git -C "$repo_dir" fetch --depth 1 origin
  if [[ -n "$branch" ]]; then
    git -C "$repo_dir" checkout "$branch"
    git -C "$repo_dir" pull --ff-only origin "$branch"
  else
    local current_branch
    current_branch="$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD)"
    git -C "$repo_dir" pull --ff-only origin "$current_branch"
  fi
}

mapfile -t entries < <(python3 - "$CONFIG_FILE" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
for item in cfg.get('models', []):
    name = item['name']
    target = item['target']
    url = item['url']
    print(f"{name}\t{target}\t{url}")
PY
)

if [[ "$INSTALL_CUSTOM_NODES" == "1" ]]; then
  mkdir -p "$CUSTOM_NODES_ROOT"

  ensure_git_repo "https://github.com/Fannovel16/comfyui_controlnet_aux.git" \
    "$CUSTOM_NODES_ROOT/comfyui_controlnet_aux"
  ensure_git_repo "https://github.com/cubiq/PuLID_ComfyUI.git" \
    "$CUSTOM_NODES_ROOT/PuLID_ComfyUI"

  install_requirements_if_exists "$CUSTOM_NODES_ROOT/comfyui_controlnet_aux/requirements.txt"

  echo "[PIP ] extra runtime deps"
  python3 -m pip install --no-cache-dir --force-reinstall --ignore-installed "numpy==2.4.4"
  python3 -m pip install --no-cache-dir --force-reinstall --ignore-installed --no-deps \
    "scipy==1.17.1" \
    "onnx==1.21.0" \
    "onnxruntime==1.24.4" \
    "onnxruntime-gpu==1.24.4" \
    "insightface==0.7.3"
  echo "[PIP ] using RunPod-aligned pins: numpy 2.4.4 / scipy 1.17.1 / onnx 1.21.0 / ort-gpu 1.24.4 / insightface 0.7.3"
fi

for row in "${entries[@]}"; do
  IFS=$'\t' read -r name target url <<<"$row"

  if [[ "$url" == *"{{CIVITAI_TOKEN}}"* ]]; then
    if [[ -z "$CIVITAI_TOKEN" ]]; then
      echo "[ERR] CIVITAI_TOKEN is empty but required for $name" >&2
      exit 1
    fi
    url="${url//\{\{CIVITAI_TOKEN\}\}/$CIVITAI_TOKEN}"
  fi

  download_file "$url" "$COMFY_ROOT/$target"
done

echo "[DONE] V16 models installed into: $COMFY_ROOT"
