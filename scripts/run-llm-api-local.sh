#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.local"

echo "🔧 Starting LLM API in LOCAL (MPS) mode..."
echo "   Using env file: $ENV_FILE"
echo

# ------------------------------
# Check env file exists
# ------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ .env.local not found at: $ENV_FILE"
    echo "   Create one or copy from .env.example"
    exit 1
fi

# ------------------------------
# Load environment variables
# ------------------------------
set -a
source "$ENV_FILE"
set +a

# ------------------------------
# Validate required vars
# ------------------------------
REQUIRED_VARS=("MODEL_ID" "MODEL_DEVICE" "DATABASE_URL" "REDIS_URL")

for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ Missing required environment variable: $var"
        exit 1
    fi
done

API_PORT="${API_PORT:-8000}"

# ------------------------------
# Summary
# ------------------------------
echo "📌 Environment summary:"
echo "   MODEL_ID       = $MODEL_ID"
echo "   MODEL_DEVICE   = $MODEL_DEVICE"
echo "   DATABASE_URL   = $DATABASE_URL"
echo "   REDIS_URL      = $REDIS_URL"
echo "   API_PORT       = $API_PORT"
echo
echo "🚀 Running FastAPI server with uv..."
echo

# ------------------------------
# Launch API
# ------------------------------
cd "$ROOT_DIR"

ENV=dev PORT="$API_PORT" uv run serve