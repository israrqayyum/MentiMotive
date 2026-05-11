#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="${PROJECT_ROOT}/.run"
BACKEND_PID_FILE="${PID_DIR}/backend.pid"
NGROK_PID_FILE="${PID_DIR}/ngrok.pid"

stop_from_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [ ! -f "$pid_file" ]; then
    echo "[INFO] No ${name} pid file found."
    return 0
  fi

  local pid
  pid="$(cat "$pid_file")"

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "[OK] Stopped ${name} (PID ${pid})."
  else
    echo "[WARN] ${name} process not running (PID ${pid})."
  fi

  rm -f "$pid_file"
}

stop_from_pid_file "ngrok" "$NGROK_PID_FILE"
stop_from_pid_file "backend" "$BACKEND_PID_FILE"

echo "[INFO] Stop routine complete."
echo "[INFO] Recent backend logs:"
if [ -f "${PROJECT_ROOT}/log/backend.log" ]; then
  tail -n 20 "${PROJECT_ROOT}/log/backend.log" || true
else
  echo "[INFO] No backend log file found."
fi

echo "[INFO] Recent ngrok logs:"
if [ -f "${PROJECT_ROOT}/log/ngrok.log" ]; then
  tail -n 20 "${PROJECT_ROOT}/log/ngrok.log" || true
else
  echo "[INFO] No ngrok log file found."
fi
