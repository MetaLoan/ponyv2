#!/usr/bin/env bash
set -euo pipefail

PID="${1:-19}"
COMFY_API_URL="${COMFY_API_URL:-http://127.0.0.1:8188}"
COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-/workspace/runpod-slim/ComfyUI/output}"
COMFY_LOG_PATH="${COMFY_LOG_PATH:-/tmp/comfy.log}"

section() {
  printf '\n===== %s =====\n' "$1"
}

safe_cmd() {
  "$@" 2>/dev/null || true
}

sample_pid_status() {
  safe_cmd grep -E 'Name|State|PPid|Threads|VmRSS|voluntary_ctxt_switches|nonvoluntary_ctxt_switches' "/proc/$PID/status"
  printf 'WCHAN='
  safe_cmd cat "/proc/$PID/wchan"
}

section "TIME"
date -u

section "PROCESSES"
safe_cmd ps -ef

section "PID $PID STATUS"
sample_pid_status

section "COMFY HEALTH"
safe_cmd curl -fsS "$COMFY_API_URL/system_stats"

section "COMFY QUEUE"
safe_cmd curl -fsS "$COMFY_API_URL/queue"

section "COMFY LOG TAIL"
safe_cmd tail -n 80 "$COMFY_LOG_PATH"

section "RECENT OUTPUTS"
safe_cmd ls -lt "$COMFY_OUTPUT_DIR"

section "SOCKET FDS PID $PID"
safe_cmd ls -l "/proc/$PID/fd"

section "SECOND SAMPLE AFTER 5S"
sleep 5
sample_pid_status
printf 'QUEUE='
safe_cmd curl -fsS "$COMFY_API_URL/queue"
echo
