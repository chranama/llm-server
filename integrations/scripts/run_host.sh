#!/usr/bin/env bash
set -euo pipefail

# Run integrations tests against an already-running stack on the host.
# Config via env:
#   INTEGRATIONS_BASE_URL (default http://localhost:8000)
#   INTEGRATIONS_MODE     (auto|generate_only|full) default auto
#   API_KEY               (optional)
#   INTEGRATIONS_TIMEOUT  (default 30)
#   PYTEST_ARGS           (optional extra args)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

need_cmd curl

export INTEGRATIONS_BASE_URL
export INTEGRATIONS_TIMEOUT
export INTEGRATIONS_MODE
: "${INTEGRATIONS_BASE_URL:=http://localhost:8000}"
: "${INTEGRATIONS_TIMEOUT:=30}"
: "${INTEGRATIONS_MODE:=auto}"

print_config

log "Waiting for /healthz..."
"${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/healthz" --timeout 60

# If you use /readyz, also wait on it (tolerant if not present)
if curl -fsS "${INTEGRATIONS_BASE_URL%/}/readyz" >/dev/null 2>&1; then
  log "Waiting for /readyz..."
  "${SCRIPT_DIR}/wait_http.sh" --base-url "${INTEGRATIONS_BASE_URL}" --path "/readyz" --timeout 120
fi

log "Running pytest in integrations/ ..."
run_pytest