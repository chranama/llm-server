#!/usr/bin/env bash
set -euo pipefail

# wait_http.sh --base-url http://localhost:8000 --path /healthz --timeout 60
# Defaults: base-url from INTEGRATIONS_BASE_URL, path=/healthz, timeout=60

BASE_URL="${INTEGRATIONS_BASE_URL:-http://localhost:8000}"
PATH_="/healthz"
TIMEOUT_S=60
SLEEP_S=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --path) PATH_="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --sleep) SLEEP_S="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

BASE_URL="${BASE_URL%/}"
URL="${BASE_URL}${PATH_}"

start="$(date +%s)"
echo "Waiting for ${URL} (timeout=${TIMEOUT_S}s)..."

while true; do
  if curl -fsS "${URL}" >/dev/null 2>&1; then
    echo "OK: ${URL}"
    exit 0
  fi

  now="$(date +%s)"
  elapsed="$((now - start))"
  if (( elapsed >= TIMEOUT_S )); then
    echo "Timed out waiting for ${URL} after ${elapsed}s" >&2
    exit 1
  fi

  sleep "${SLEEP_S}"
done