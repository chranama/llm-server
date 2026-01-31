#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

need_cmd kubectl

ROOT="$(repo_root)"
: "${K8S_OVERLAY:=local-generate-only}"
: "${K8S_CONTEXT:=}"
: "${K8S_NAMESPACE:=}"  # usually declared in kustomize; can be blank

KUBECTL=(kubectl)
if [[ -n "${K8S_CONTEXT}" ]]; then
  KUBECTL+=(--context "${K8S_CONTEXT}")
fi

OVERLAY_DIR="${ROOT}/deploy/k8s/overlays/${K8S_OVERLAY}"
[[ -d "${OVERLAY_DIR}" ]] || die "Overlay not found: ${OVERLAY_DIR} (set K8S_OVERLAY=...)"

log "Applying kustomize overlay: ${OVERLAY_DIR}"
"${KUBECTL[@]}" apply -k "${OVERLAY_DIR}"

log "Applied overlay ok."