#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

need_cmd kubectl
need_cmd curl

: "${INTEGRATIONS_MODE:=full}"
: "${INTEGRATIONS_TIMEOUT:=30}"
export INTEGRATIONS_MODE INTEGRATIONS_TIMEOUT

: "${K8S_APPLY:=0}"
: "${K8S_OVERLAY:=prod-gpu-full}"
: "${K8S_CONTEXT:=}"

# From service.yaml
: "${K8S_NAMESPACE:=llm}"
: "${K8S_SERVICE_NAME:=api}"
: "${K8S_REMOTE_PORT:=8000}"
: "${K8S_LOCAL_PORT:=8000}"

if [[ "${K8S_APPLY}" == "1" ]]; then
  K8S_OVERLAY="${K8S_OVERLAY}" K8S_CONTEXT="${K8S_CONTEXT}" "${SCRIPT_DIR}/k8s_apply_overlay.sh"
fi

if [[ -n "${INTEGRATIONS_BASE_URL:-}" ]]; then
  INTEGRATIONS_BASE_URL="$(normalize_base_url "${INTEGRATIONS_BASE_URL}")"
  export INTEGRATIONS_BASE_URL
  print_config

  log "Waiting for /healthz..."
  "${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/healthz" --timeout 300

  log "Running pytest (k8s full, direct URL)..."
  run_pytest
  exit 0
fi

KUBECTL=(kubectl)
if [[ -n "${K8S_CONTEXT}" ]]; then
  KUBECTL+=(--context "${K8S_CONTEXT}")
fi

PF_LOG="$(mktemp -t k8s-portforward.XXXXXX.log)"
PF_PID=""

cleanup() {
  if [[ -n "${PF_PID}" ]]; then
    log "Stopping port-forward (pid=${PF_PID})..."
    kill "${PF_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${PF_LOG}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "Port-forwarding svc/${K8S_SERVICE_NAME} ${K8S_LOCAL_PORT}:${K8S_REMOTE_PORT} (ns=${K8S_NAMESPACE})..."
"${KUBECTL[@]}" -n "${K8S_NAMESPACE}" port-forward "svc/${K8S_SERVICE_NAME}" "${K8S_LOCAL_PORT}:${K8S_REMOTE_PORT}" >"${PF_LOG}" 2>&1 &
PF_PID="$!"

sleep 1
if ! kill -0 "${PF_PID}" >/dev/null 2>&1; then
  log "Port-forward failed. Logs:"
  sed -n '1,200p' "${PF_LOG}" >&2 || true
  exit 1
fi

export INTEGRATIONS_BASE_URL="http://localhost:${K8S_LOCAL_PORT}"
print_config

log "Waiting for /healthz..."
"${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/healthz" --timeout 300

if curl -fsS "${INTEGRATIONS_BASE_URL%/}/readyz" >/dev/null 2>&1; then
  log "Waiting for /readyz..."
  "${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/readyz" --timeout 300
fi

log "Running pytest (k8s full)..."
run_pytest