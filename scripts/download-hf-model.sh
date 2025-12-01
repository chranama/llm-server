#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "📦 Hugging Face Model Downloader"
echo

# -----------------------------------------------------
# Auto-load .env if HF_TOKEN is missing
# -----------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ".env" ]]; then
    echo "ℹ️  HF_TOKEN not found in environment. Loading from .env ..."
    export $(grep -v '^#' .env | xargs)
  fi
fi

# -----------------------------------------------------
# Validate HF_TOKEN after auto-load
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
# We *ignore* HF_HOME from .env (which is for the container: /root/.cache/huggingface)
# and default to the current user's cache on the host.
if [[ -z "${HF_HOME:-}" || "$HF_HOME" == "/root/.cache/huggingface" ]]; then
  export HF_HOME="$HOME/.cache/huggingface"
fi

# Allow explicit override if the user really wants something else:
# HF_HOME=/some/path ./scripts/download-hf-model.sh
export HF_HOME

# -----------------------------------------------------
# Model selection
# -----------------------------------------------------
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
export MODEL_ID  # for Python subprocess

echo "➡️  MODEL_ID = $MODEL_ID"
echo "➡️  HF_HOME  = $HF_HOME"
echo

# -----------------------------------------------------
# Download the model using Python + HF transformers
# -----------------------------------------------------
python - << 'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
hf_home = os.environ.get("HF_HOME")

print(f"🔥 Downloading snapshot for: {model_id}")
print(f"📂 Cache location (HF_HOME): {hf_home}")

# This will populate the HF cache without instantiating the model class.
# It respects HF_HOME and HF_TOKEN from the environment.
local_dir = snapshot_download(
    repo_id=model_id,
    local_dir=None,          # use HF cache layout
    local_dir_use_symlinks=True,
)

print(f"✅ Snapshot downloaded into cache: {local_dir}")
PY