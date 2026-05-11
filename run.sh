#!/usr/bin/env bash
set -euo pipefail

APP_PORT=8000
NGROK_DOMAIN="cabbage-regroup-outright.ngrok-free.dev"
NGROK_API="http://127.0.0.1:4040/api/tunnels"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/log"
PID_DIR="${PROJECT_ROOT}/.run"
BACKEND_PID_FILE="${PID_DIR}/backend.pid"
NGROK_PID_FILE="${PID_DIR}/ngrok.pid"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*" >&2; }

require() {
  command -v "$1" >/dev/null 2>&1 || { fail "'$1' is required but not in PATH."; exit 1; }
}

cleanup_on_interrupt() {
  echo
  info "Ctrl+C detected. Stopping services..."
  "${PROJECT_ROOT}/stop.sh"
  exit 0
}

is_port_open() {
  curl -fsS "http://127.0.0.1:${APP_PORT}/docs" >/dev/null 2>&1
}

mkdir -p "$LOG_DIR" "$PID_DIR"
cd "$PROJECT_ROOT"

require curl
require ngrok

trap cleanup_on_interrupt INT TERM

if is_port_open; then
  warn "Backend is already running on port ${APP_PORT}. Reusing existing process."
else
  info "Starting backend on port ${APP_PORT}..."
  if [ -x "${PROJECT_ROOT}/venv/Scripts/python.exe" ]; then
    nohup "${PROJECT_ROOT}/venv/Scripts/python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port "${APP_PORT}" > "${LOG_DIR}/backend.log" 2>&1 &
  else
    nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port "${APP_PORT}" > "${LOG_DIR}/backend.log" 2>&1 &
  fi
  echo "$!" > "$BACKEND_PID_FILE"
fi

info "Waiting for backend readiness..."
for _ in {1..45}; do
  if is_port_open; then
    ok "Backend is reachable at http://localhost:${APP_PORT}"
    break
  fi
  sleep 1
done

if ! is_port_open; then
  fail "Backend did not start. Check ${LOG_DIR}/backend.log"
  exit 1
fi

if [ -f "$NGROK_PID_FILE" ] && kill -0 "$(cat "$NGROK_PID_FILE")" 2>/dev/null; then
  warn "ngrok is already running from this script."
else
  info "Starting ngrok tunnel with reserved domain ${NGROK_DOMAIN}..."
  nohup ngrok http "${APP_PORT}" --domain="${NGROK_DOMAIN}" --log=stdout > "${LOG_DIR}/ngrok.log" 2>&1 &
  echo "$!" > "$NGROK_PID_FILE"
fi

info "Waiting for ngrok tunnel..."
PUBLIC_URL=""
for _ in {1..30}; do
  PUBLIC_URL="$(curl -fsS "$NGROK_API" 2>/dev/null | sed -n 's/.*"public_url":"\([^"]*\)".*/\1/p' | head -n 1 || true)"
  [ -n "$PUBLIC_URL" ] && break
  sleep 1
done

touch "${LOG_DIR}/backend.log" "${LOG_DIR}/ngrok.log"

echo
printf "${GREEN}=====================================================${NC}\n"
printf "${GREEN}  Service startup complete${NC}\n"
printf "${GREEN}=====================================================${NC}\n"
printf "  Backend (local):    http://localhost:${APP_PORT}\n"
printf "  Backend (public):   %s\n" "${PUBLIC_URL:-${NGROK_DOMAIN}}"
printf "  Docs (local):       http://localhost:${APP_PORT}/docs\n"
printf "  Docs (public):      https://${NGROK_DOMAIN}/docs\n"
printf "  Ngrok dashboard:    http://localhost:4040\n"
printf "${GREEN}=====================================================${NC}\n"
printf "  Live logs start below. Press Ctrl+C to stop all services.\n"
printf "${GREEN}=====================================================${NC}\n"

tail -n 30 -f "${LOG_DIR}/backend.log" "${LOG_DIR}/ngrok.log"