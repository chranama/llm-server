#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

need_cmd docker
need_cmd curl

ROOT="$(repo_root)"

# NEW: prefer your deploy/compose path by default
: "${COMPOSE_FILE_PATH:=${ROOT}/deploy/compose/docker-compose.yml}"

find_compose_file() {
  local cand
  for cand in \
    "${COMPOSE_FILE_PATH:-}" \
    "${ROOT}/docker-compose.yml" \
    "${ROOT}/docker-compose.yaml" \
    "${ROOT}/compose.yml" \
    "${ROOT}/compose.yaml"
  do
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

COMPOSE_FILE="$(find_compose_file || true)"
[[ -n "${COMPOSE_FILE}" ]] || die "Could not find compose file. Set COMPOSE_FILE_PATH=/path/to/compose.yaml"

: "${COMPOSE_PROJECT_NAME:=llm_server_itest_full}"
: "${INTEGRATIONS_BASE_URL:=http://localhost:8000}"
: "${INTEGRATIONS_TIMEOUT:=30}"
: "${INTEGRATIONS_MODE:=full}"

export INTEGRATIONS_BASE_URL INTEGRATIONS_TIMEOUT INTEGRATIONS_MODE

print_config
log "Compose file: ${COMPOSE_FILE}"
log "Compose project: ${COMPOSE_PROJECT_NAME}"

compose() {
  # shellcheck disable=SC2086
  docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT_NAME}" ${COMPOSE_PROFILES:+--profile "${COMPOSE_PROFILES}"} "$@"
}

cleanup() {
  log "Tearing down compose stack..."
  compose down -v --remove-orphans || true
}
trap cleanup EXIT

log "Bringing up compose stack (full)..."
compose up -d --remove-orphans

log "Waiting for /healthz..."
"${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/healthz" --timeout 120

if curl -fsS "${INTEGRATIONS_BASE_URL%/}/readyz" >/dev/null 2>&1; then
  log "Waiting for /readyz..."
  "${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/readyz" --timeout 180
fi

log "Running pytest (full)..."
run_pytest