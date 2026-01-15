#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "📦 Hugging Face Model Downloader"
echo

# -----------------------------------------------------
# Auto-load .env ONLY for HF_TOKEN (avoid clobbering MODEL_ID)
# -----------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ".env" ]]; then
    echo "ℹ️  HF_TOKEN not found in environment. Loading HF_TOKEN from .env ..."
    # Extract HF_TOKEN line only (supports optional quotes)
    HF_TOKEN_LINE="$(grep -E '^[[:space:]]*HF_TOKEN=' .env | tail -n 1 || true)"
    if [[ -n "$HF_TOKEN_LINE" ]]; then
      # shellcheck disable=SC2163
      export "$HF_TOKEN_LINE"
      # Remove surrounding quotes if present (export keeps them sometimes)
      HF_TOKEN="${HF_TOKEN%\"}"; HF_TOKEN="${HF_TOKEN#\"}"
      HF_TOKEN="${HF_TOKEN%\'}"; HF_TOKEN="${HF_TOKEN#\'}"
      export HF_TOKEN
    fi
  fi
fi

# -----------------------------------------------------
# Validate HF_TOKEN
# -----------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "❌ HF_TOKEN is still not set."
  echo "   Please ensure .env contains HF_TOKEN or export it manually:"
  echo "       export HF_TOKEN=hf_...your_token..."
  exit 1
fi

echo "🔐 HF_TOKEN loaded successfully."
echo

# -----------------------------------------------------
# HF_HOME: force host cache by default
# -----------------------------------------------------
# Ignore container HF_HOME default
if [[ -z "${HF_HOME:-}" || "$HF_HOME" == "/root/.cache/huggingface" ]]; then
  export HF_HOME="$HOME/.cache/huggingface"
fi

# Allow explicit override:
# HF_HOME=/some/path ./scripts/download-hf-model.sh <model_id>
export HF_HOME

# -----------------------------------------------------
# Model selection (CLI arg wins; env MODEL_ID next; fallback default)
# -----------------------------------------------------
if [[ $# -ge 1 ]]; then
  MODEL_ID="$1"
else
  MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
fi
export MODEL_ID

echo "➡️  MODEL_ID = $MODEL_ID"
echo "➡️  HF_HOME  = $HF_HOME"
echo

# -----------------------------------------------------
# Download the model using Python + HF Hub
# -----------------------------------------------------
python - << 'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
hf_home = os.environ.get("HF_HOME")
token = os.environ.get("HF_TOKEN")

print(f"🔥 Downloading snapshot for: {model_id}")
print(f"📂 Cache location (HF_HOME): {hf_home}")

local_dir = snapshot_download(
    repo_id=model_id,
    local_dir=None,          # use HF cache layout
    local_dir_use_symlinks=True,
    token=token,
)

print(f"✅ Snapshot downloaded into cache: {local_dir}")
PY

echo
echo "🎉 Done."