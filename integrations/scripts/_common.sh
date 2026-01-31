#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Common helpers for integrations/scripts/*
# -------------------------

log() { echo "[$(date +'%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# Find repo root (prefers git).
repo_root() {
  if command -v git >/dev/null 2>&1; then
    local r
    r="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "${r}" ]]; then
      echo "${r}"
      return 0
    fi
  fi

  # Fallback: walk up from this script location until we find "integrations/"
  local d
  d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  echo "${d}"
}

integrations_dir() {
  echo "$(repo_root)/integrations"
}

# Print key configuration once at runtime
print_config() {
  log "Repo root: $(repo_root)"
  log "Integrations dir: $(integrations_dir)"
  log "INTEGRATIONS_BASE_URL=${INTEGRATIONS_BASE_URL:-<unset>}"
  log "INTEGRATIONS_MODE=${INTEGRATIONS_MODE:-<unset>}"
  log "API_KEY=${API_KEY:+<set>}${API_KEY:-<unset>}"
  log "INTEGRATIONS_TIMEOUT=${INTEGRATIONS_TIMEOUT:-<unset>}"
}

# Run pytest within integrations/ using uv (no global pyproject required)
run_pytest() {
  need_cmd uv

  local idir
  idir="$(integrations_dir)"
  [[ -f "${idir}/pyproject.toml" ]] || die "Expected ${idir}/pyproject.toml"

  (
    cd "${idir}"
    # shellcheck disable=SC2086
    uv run pytest ${PYTEST_ARGS:-}
  )
}

# Normalize base url (strip trailing slash)
normalize_base_url() {
  local u="${1:-}"
  u="${u%/}"
  echo "${u}"
}